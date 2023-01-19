import json
import logging
import os
import random
import time
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from tqdm import tqdm, trange
from torch.cuda.amp import autocast as autocast, GradScaler
from bert4pytorch.trainers.utils import PGD, FGM
from bert4pytorch.trainers.metrics import compute_cls_metrics
from bert4pytorch.optimizers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from bert4pytorch.trainers.logger_utils import logger
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, train_data, args, eval_data=None, test_data=None, metrics=None,
                 use_loss='ce', label_list=None, train_collate_fn=None, eval_collate_fn=None, use_tqdm=True,
                 num_workers=0, sampler=None, train_logging_every=-1, gradient_accumulation_steps=1,
                 warmup_steps=0.1, weight_decay=1e-3,shuffle=True,show_train_result=True,
                 validate_every=-1, callbacks=None, fp16=False, use_adv=None,
                 ):
        super(Trainer, self).__init__()

        self.args = args
        self.use_loss = use_loss
        self.label_list = label_list
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.metrics = metrics
        self.fp16 = fp16
        self.use_adv = use_adv
        self.sampler = sampler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validate_every = validate_every
        self.train_logging_every = train_logging_every
        self.show_train_result = show_train_result

        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data
        self.use_tqdm = use_tqdm
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        if self.label_list is not None:
            self.num_labels = len(self.label_list)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.to(self.device)
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.best_loss = 9999
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")
        if isinstance(self.train_data, Dataset):
            self.train_data_iterator = torch.utils.data.DataLoader(
                dataset=self.train_data, batch_size=self.args.batch_size, sampler=self.sampler,
                num_workers=num_workers, collate_fn=train_collate_fn,shuffle=shuffle
            )
        else:
            raise TypeError("train_data type {} not support".format(type(train_data)))

        if eval_data is not None:
            if isinstance(self.eval_data, Dataset):
                self.eval_data_iterator = torch.utils.data.DataLoader(
                    dataset=self.eval_data, batch_size=self.args.batch_size, sampler=self.sampler,
                    num_workers=num_workers, collate_fn=eval_collate_fn,shuffle=shuffle
                )
            else:
                raise TypeError("eval_data type {} not support".format(type(train_data)))

        if test_data is not None:
            if isinstance(self.test_data, Dataset):
                self.test_data_iterator = torch.utils.data.DataLoader(
                    dataset=self.test_data, batch_size=self.args.batch_size, sampler=self.sampler,
                    num_workers=num_workers, collate_fn=eval_collate_fn,shuffle=shuffle
                )
            else:
                raise TypeError("test_data type {} not support".format(type(train_data)))
        assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be no less than 1."
        if not (self.args.save_path is None or isinstance(self.args.save_path, str)):
            raise ValueError("save_path can only be None or `str`.")


    def compute_loss(self, is_eval, pred, label=None, use_loss=None, input=None):
        loss_fct = None
        if use_loss == None:
            use_loss = 'ce'

        if use_loss == 'ce':
            loss_fct = nn.CrossEntropyLoss()
        elif use_loss == 'focal_loss':
            pass

        loss = loss_fct(pred.view(-1, self.num_labels), label.view(-1))
        return loss

    def evaluate_for_batch(self, model, batch):
        outputs = model(**batch)
        return outputs


    def evaluate_metrics(self, is_eval, outputs, labels=None, input=None):
        labels = labels.detach().cpu().numpy()
        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        results = compute_cls_metrics(preds, labels)
        return results


    def train(self):
        total_results = {}
        t_total = len(self.train_data_iterator) // self.gradient_accumulation_steps * self.args.num_train_epochs

        if self.fp16:
            scaler = GradScaler()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=1e-8)

        if not self.warmup_steps:
            self.warmup_steps = int(t_total * 0.05)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_data))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        if self.use_tqdm:
           train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        else:
            train_iterator = range(self.args.num_train_epochs)

        last_epoch = None
        for i, _ in enumerate(train_iterator):
            logger.info("  Current epoch = %d", i)
            for step, batch in enumerate(self.train_data_iterator):
                self.model.train()

                batch_device = {}
                for name, item in batch.items():
                    try:
                        item = item.to(self.device)
                        batch_device[name] = item
                    except AttributeError:
                        batch_device[name] = item

                batch_x  = {}
                batch_y = None
                train_has_label = False
                for name, item in batch.items():
                    if name == 'label':
                        try:
                            batch_y = batch['label'].to(self.device)
                        except AttributeError:
                            batch_y = batch['label']
                        train_has_label = True
                    else:
                        try:
                            batch_x[name] = item.to(self.device)
                        except AttributeError:
                            batch_x[name] = item


                if train_has_label == False:
                    batch_y = None

                if self.fp16:
                    with autocast():
                        outputs = self.model(**batch_device)
                else:
                    outputs = self.model(**batch_device)

                loss = self.compute_loss(is_eval=False, pred=outputs, label=batch_y,
                                         use_loss=self.use_loss, input=batch_device)

                if self.train_logging_every != -1 or self.show_train_result:
                    if batch_y != None:
                        results = self.evaluate_metrics(is_eval=False, outputs=outputs, labels=batch_y, input=batch_device)
                    else:
                        results = self.evaluate_metrics(is_eval=False, outputs=outputs, input=batch_device)

                    if results != None:
                        for name, value in results.items():
                            if name not in total_results:
                                total_results[name] = []
                            total_results[name].append(value)


                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.use_adv == 'pgd':
                    pgd = PGD(model=self.model)
                    pgd_k = 3
                    pgd.backup_grad()
                    for _t in range(pgd_k):
                        pgd.attack(is_first_attack=(_t == 0))
                        if _t != pgd_k - 1:
                            self.model.zero_grad()
                        else:
                            pgd.restore_grad()

                        if self.fp16:
                            with autocast():
                                outputs = self.model(**batch_device)
                        else:
                            outputs = self.model(**batch_device)
                        logits = outputs
                        loss_adv = self.compute_loss(is_eval=False, pred=logits, label=batch_y, use_loss=self.use_loss, input=batch_device)

                        if self.fp16:
                            scaler.scale(loss_adv).backward()
                        else:
                            loss_adv.backward()
                    pgd.restore()

                if self.use_adv == 'fgm':
                    fgm = FGM(model=self.model)
                    fgm.attack()

                    if self.fp16:
                        with autocast():
                            outputs = self.model(**batch_device)
                    else:
                        outputs = self.model(**batch_device)
                    logits = outputs
                    loss_adv = self.compute_loss(is_eval=False, pred=logits, label=batch_y, use_loss=self.use_loss, input=batch_device)

                    if self.fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward()
                    fgm.restore()

                tr_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    if self.fp16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.eval_data is None:
                        if self.train_logging_every == -1:
                            raise TypeError('Please set the logging every for training!')
                        else:
                            if self.train_logging_every > 0 and (
                                    global_step % self.train_logging_every == 0 or global_step == t_total):
                                metrics_results = {}
                                metrics_results['loss'] = round(tr_loss / self.train_logging_every, 7)
                                tr_loss = 0.

                                if total_results != []:
                                    for name, value_list in total_results.items():
                                        if name not in metrics_results:
                                            metrics_results[name] = round(sum(value_list) / len(value_list), 6)

                                logger.info("***** Train results *****")
                                for k, v in metrics_results.items():
                                    logger.info("{}: {}".format(k, v))
                                self.save_model()

                    if self.eval_data is not None:
                        if self.validate_every > 0 and (
                                global_step % self.validate_every == 0 or global_step == t_total):

                            if global_step == self.validate_every:
                                logger.info("***** Running evaluation  *****")
                                logger.info("  Num examples = %d", len(self.eval_data))


                            if self.show_train_result:
                                metrics_results = {}
                                metrics_results['loss'] = round(tr_loss / self.validate_every, 7)
                                tr_loss = 0.

                                if total_results != []:
                                    for name, value_list in total_results.items():
                                        if name not in metrics_results:
                                            metrics_results[name] = round(sum(value_list) / len(value_list), 6)

                                logger.info("***** Train results *****")
                                for k, v in metrics_results.items():
                                    logger.info("{}: {}".format(k, v))


                            eval_results = self.evaluate()
                            eval_current_score = eval_results.get(self.metric_key_for_early_stop, -1)
                            if eval_current_score == -1:
                                raise TypeError('Please type the right metric_key_for_early_stop')

                            if 'loss' in self.metric_key_for_early_stop:
                                eval_current_loss = eval_current_score
                                if eval_current_loss < self.best_loss:
                                    eval_best_loss = eval_results.get(self.metric_key_for_early_stop, -1)

                                    self.best_loss = eval_best_loss
                                    self.early_stopping_counter = 0
                                    self.save_model()
                                else:
                                    self.early_stopping_counter += 1
                                    if self.early_stopping_counter >= self.patience:
                                        self.do_early_stop = True
                                        logger.info(
                                            "the loss has reached the patience of {}".format(self.patience))
                                        logger.info("lowest loss is {}".format(self.best_loss))

                            if 'loss' not in self.metric_key_for_early_stop:
                                if eval_current_score > self.best_score:
                                    eval_best_score = eval_results.get(self.metric_key_for_early_stop, -1)
                                    if eval_current_score == -1:
                                        raise TypeError('Please type the right metric_key_for_early_stop')
                                    self.best_score = eval_best_score
                                    self.early_stopping_counter = 0
                                    self.save_model()
                                else:
                                    self.early_stopping_counter += 1
                                    if self.early_stopping_counter >= self.patience:
                                        self.do_early_stop = True
                                        logger.info(
                                            "the score has reached the patience of {}".format(self.patience))
                                        logger.info("best score is {}".format(self.best_score))

                                if self.do_early_stop:
                                    break
                if self.do_early_stop:
                    break
            if self.do_early_stop:
                break
            last_epoch = i
        logger.info('Model end at epoch {}'.format(last_epoch))
        return


    def evaluate(self):

        eval_loss = 0.0
        nb_eval_steps = 0
        total_results = {}

        self.model.eval()
        for batch in self.eval_data_iterator:
            with torch.no_grad():

                batch_device = {}
                for name, item in batch.items():
                    try:
                        item = item.to(self.device)
                        batch_device[name] = item
                    except AttributeError:
                        batch_device[name] = item

                batch_x = {}
                batch_y = None
                for name, item in batch.items():
                    if name == 'label':
                        try:
                            batch_y = batch['label'].to(self.device)
                        except AttributeError:
                            batch_y = batch['label']
                    else:
                        try:
                            batch_x[name] = item.to(self.device)
                        except AttributeError:
                            batch_x[name] = item

                outputs = self.evaluate_for_batch(self.model, batch_device)

                tmp_eval_loss = self.compute_loss(is_eval=True, pred=outputs, label=batch_y,
                                                  use_loss=self.use_loss, input=batch_device)
                if tmp_eval_loss != None:
                    eval_loss += tmp_eval_loss.mean().item()

            if batch_y != None:
                results = self.evaluate_metrics(is_eval=True, outputs=outputs, labels=batch_y, input=batch_device)
            else:
                results = self.evaluate_metrics(is_eval=True, outputs=outputs, input=batch_device)

            if results != None:
                for name, value in results.items():
                    if name not in total_results:
                        total_results[name] = []
                    total_results[name].append(value)

            nb_eval_steps += 1


        eval_loss = eval_loss / nb_eval_steps

        metrics_results = {}
        metrics_results['loss'] = round(eval_loss,7)

        if total_results != []:
            for name, value_list in total_results.items():
                if name not in metrics_results:
                    metrics_results[name] = round(sum(value_list) / len(value_list), 6)


        logger.info("***** Eval results *****")
        for k,v in metrics_results.items():
            logger.info("{}: {}".format(k, v))

        return metrics_results


    def test(self):
        logger.info("***** Running testing  *****")
        logger.info("  Num examples = %d", len(self.test_data))

        self.load_model()
        self.model.eval()
        test_loss = 0.0
        nb_test_steps = 0
        test_has_label = False
        total_results = {}


        for batch in tqdm(self.test_data_iterator, desc="Testing", nrows=20):
            with torch.no_grad():
                batch_device = {}
                for name, item in batch.items():
                    try:
                        item = item.to(self.device)
                        batch_device[name] = item
                    except AttributeError:
                        batch_device[name] = item

                batch_y = None
                for name, item in batch.items():
                    if name == 'label':
                        try:
                            batch_y = batch['label'].to(self.device)
                        except AttributeError:
                            batch_y = batch['label']
                        test_has_label = True

                if test_has_label == False:
                    batch_y = None

                outputs = self.evaluate_for_batch(self.model, batch_device)

                tmp_test_loss = self.compute_loss(is_eval=True, pred=outputs, label=batch_y,
                                                  use_loss=self.use_loss, input=batch_device)
                if tmp_test_loss!=None:
                    test_loss += tmp_test_loss.mean().item()

            if batch_y != None:
                results = self.evaluate_metrics(is_eval=True, outputs=outputs, labels=batch_y, input=batch_device)
            else:
                results = self.evaluate_metrics(is_eval=True, outputs=outputs, input=batch_device)

            if results != None:
                for name, value in results.items():
                    if name not in total_results:
                        total_results[name] = []
                    total_results[name].append(value)

            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps

        metrics_results = {}
        metrics_results['loss'] = round(test_loss,7)

        if total_results != []:
            for name, value_list in total_results.items():
                if name not in metrics_results:
                    metrics_results[name] = round(sum(value_list) / len(value_list), 6)


        logger.info("***** Test results *****")
        for k,v in metrics_results.items():
            logger.info("{}: {}".format(k, v))

        return metrics_results

    def predict(self, output_file=None):
        logger.info("***** Running predicting  *****")
        logger.info("  Num examples = %d", len(self.test_data))

        self.load_model()

        preds = None
        self.model.eval()
        for batch in tqdm(self.test_data_iterator, desc="Predicting", nrows=20):
            with torch.no_grad():

                batch_x = {}
                for name, item in batch.items():
                    if name != 'label':
                        batch_x[name] = item.to(self.device)

                outputs = self.model(**batch_x)
                logits = outputs
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        logits = preds
        preds = np.argmax(preds, axis=1)

        list_preds = preds.tolist()
        pred_label = [self.label_list[pred_label_id] for pred_label_id in list_preds]
        if output_file is not None:
            if 'txt' in output_file:
                f_out = open(output_file, "w", encoding="utf-8")
                f_out.write("id,label" + "\n")

                for i, pred_label_id in enumerate(list_preds):
                    pred_label_name = self.label_list[pred_label_id]
                    f_out.write("%s,%s" % (str(i), str(pred_label_name)) + "\n")

            elif 'csv' in output_file:
                df = pd.DataFrame(columns=['id', 'pred_label'])
                idxs, pred_labels = [], []
                for idx, pred_label_id in enumerate(list_preds):
                    single_pred_label = self.label_list[pred_label_id]
                    idxs.append(idx)
                    pred_labels.append(single_pred_label)
                df['id'] = idxs
                df['pred_label'] = pred_labels
                df.to_csv(output_file, index=False)

        return logits, pred_label

    def save_model(self):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        state_dict = model_to_save.state_dict()
        output_model_file = os.path.join(self.args.save_path, "model.bin")
        torch.save(state_dict, output_model_file)
        torch.save(self.args, os.path.join(self.args.save_path, 'training_args.bin'))


    def load_model(self):
        if not os.path.exists(self.args.save_path):
            raise Exception("Model doesn't exists! Train first!")

        try:
            output_model_file = os.path.join(self.args.save_path, "model.bin")
            self.model.load_state_dict(torch.load(output_model_file, map_location=self.device))

            logger.info("***** Saved model loaded! *****")
        except:
            raise Exception("Some models files might be missing...")

