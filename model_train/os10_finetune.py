#! -*- coding: utf-8 -*-

import os
import sys
import json
import torch.utils.checkpoint
sys.path.insert(0, '../')
from bert4pytorch.configs.configuration_bert import BertConfig
from bert4pytorch.tokenizers.tokenization_bert import BertTokenizer
from bert4pytorch.trainers.train_func import Trainer
from bert4pytorch.snippets import set_seed, sequence_padding, DataGenerator, DataReader
from bert4pytorch.models.model_building import *
from data_lib import *

class FinetuningArguments:
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512

    metric_key_for_early_stop = "accuracy"
    patience = 8

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'

    dataset_name = "tnews"
    data_dir = "../../code/data/tnews/"

    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/tnews/"
    os.makedirs(model_outpath, exist_ok=True)

    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"
    os.makedirs(save_path, exist_ok=True)

args = FinetuningArguments()
set_seed(args.SEED)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")
print('tokenizer:', tokenizer)

total_labels = set()

train_examples = load_data_tnews('train.json')
dev_examples = load_data_tnews('dev.json')
test_examples = load_data_tnews('dev.json')

label_map, num_labels = {l:i for i,l in enumerate(list(total_labels))}, len(list(total_labels))
print('train_examples:', len(train_examples))


train_dataset, dev_dataset, test_dataset = MyDataset_tnews(train_examples), MyDataset_tnews(dev_examples), MyDataset_tnews(test_examples)

kwargs = {"with_pool": True, "num_labels": num_labels}
config = BertConfig.from_pretrained(args.model_path, **kwargs)
model = Model_tnews(config=config, model_path=args.model_path, model_name=args.model_name)
trainer = Trainer(model, train_dataset, args, eval_data=dev_dataset, test_data=test_dataset,
                  label_list=list(label_map.keys()), validate_every=5)
trainer.train()
print('saving model...')
output_model_base_weight =  os.path.join(args.model_outpath, "model_base.bin")
output_model_task_weight =  os.path.join(args.model_outpath, "model_task.bin")

save_model(model.bert, output_model_base_weight)
save_model(model.task, output_model_task_weight)