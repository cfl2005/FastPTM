#!/usr/bin/env python3
# coding:utf-8

import time
import deepspeed
import torch.utils.checkpoint
import sys
import json
sys.path.insert(0, '../')
from bert4pytorch.models.model_building import *
import torch
from transformers import BertTokenizer
import torch
import pandas as pd
def load_base(model_pth):
   # model_pth = '/mnt/ssd/mnt/models/tnews/tnews/model_base.pth'
    tokenizer_path = '/mnt/ssd/mnt/models/bert-chinese'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = torch.load(model_pth)
    return model
if __name__ == '__main__':
    model_pth = '/mnt/ssd/mnt/models/tnews/tnews/model_base.pth'
    #model_pth ='/mnt/ssd/mnt/models/pytorch_model.bin'
    #开始加载
    time_start=time.time()
    model=load_base(model_pth)
    print(model)
    time_end = time.time()
    print('全时间',time_end-time_start)
    #加载结束base
    weight_path='/mnt/ssd/mnt/models/model_base.bin'
    #weight_path = '/mnt/ssd/mnt/models/tnews/tnews/model_base.pth'
    time_start = time.time()
    weight = torch.load(weight_path)
    model.bert.load_state_dict(weight)
    time_end = time.time()
    print('换权重',time_end-time_start)
    #