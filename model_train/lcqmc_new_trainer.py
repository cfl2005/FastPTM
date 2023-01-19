#! -*- coding: utf-8 -*-
import os
import sys
import torch.nn as nn
import torch.utils.checkpoint
sys.path.insert(0, '../')
from bert4pytorch.configs.configuration_bert import BertConfig
from bert4pytorch.tokenizers.tokenization_bert import BertTokenizer
from bert4pytorch.trainers.train_func import Trainer
from bert4pytorch.snippets import set_seed, sequence_padding, DataGenerator
from bert4pytorch.models.model_building import *
from data_lib import *

class FinetuningArguments:
    SEED = 42
    learning_rate = 4e-4
    num_train_epochs = 1
    max_len = 128
    batch_size = 512

    metric_key_for_early_stop = "accuracy"
    patience = 8

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'

    dataset_name = "lcqmc"
    data_dir = "../../code/data/lcqmc/"

    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/lcqmc/"
    os.makedirs(model_outpath, exist_ok=True)

    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"
    os.makedirs(save_path, exist_ok=True)

args = FinetuningArguments()
set_seed(args.SEED)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")
train_examples = load_data_lcqmc('train.txt', args.data_dir)
dev_examples = load_data_lcqmc('dev.txt', args.data_dir)
test_examples = load_data_lcqmc('test.txt', args.data_dir)

label_map, num_labels = {l:i for i,l in enumerate(list(total_labels))}, len(list(total_labels))

train_dataset, dev_dataset, test_dataset = MyDataset_lcqmc(train_examples), MyDataset_lcqmc(dev_examples), MyDataset_lcqmc(test_examples)

kwargs = {"num_labels": num_labels}
config = BertConfig.from_pretrained(args.model_path, **kwargs)
model = Model_lcqmc(config=config, model_path=args.model_path, model_name=args.model_name)
trainer = Trainer(model, train_dataset, args, eval_data=dev_dataset, test_data=test_dataset,
                  label_list=list(label_map.keys()), validate_every=10)#, use_adv='pgd', fp16=True)
trainer.train()
print('saving model...')
output_model_base_weight = os.path.join(args.model_outpath, "model_base.bin")
output_model_task_weight = os.path.join(args.model_outpath, "model_task.bin")

save_model(model.bert, output_model_base_weight)
save_model(model.task, output_model_task_weight)