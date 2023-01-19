#!/usr/bin/env python3
#coding:utf-8



import os
import sys
import time
import json
import re
import numpy as np
import copy

sys.path.insert(0, '../')
from bert4pytorch.snippets import set_seed, sequence_padding, DataGenerator

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue

from gpu_monitor import *
total_labels = set()

def txt2keyline(txt):
    txt = '(%s)' % txt
    txt = txt.replace('\n', ',\n')
    keyline = eval(txt)
    return keyline

def single_model_score(keyline):
    scores = {}
    scores['base-model'] = keyline[2][2]
    scores['sub-model'] = keyline[4][2]
    scores['load_base_time'] = (keyline[2][0]-keyline[1][0]) * 1000
    scores['load_sub_time'] = (keyline[4][0]-keyline[3][0]) * 1000
    scores['full_time'] = (keyline[-1][0]-keyline[0][0]) * 1000
    return scores
def calc_change_etime(keyline):
    ret = {}
    for t, k, m, mid in keyline:
        if k in ['change_model_base', 'end_model_task']:
            if not mid in ret.keys():
                ret[mid] = []
            ret[mid].append(t)
    if not ret: return 0
    change_times = []
    for k, v in ret.items():
        datv = list(map(lambda x:x[1]-x[0], zip(v[0::2], v[1::2])))
        change_times.extend(datv)
    
    if change_times==[]: return 0
    
    print('Model switching time:%s'% change_times)
    ave_time = np.average(change_times)
    print('Average time for model switching:%.4f'%ave_time)
    return ave_time

def create_monitor():
    obj = GPU_MEM(interval=0.2)
    mem = obj.get_memory()
    print('GPU Memory: %.4fMB' % mem)
    return obj

def create_log_path(logpath='logs/'):
    outpath = os.path.join(logpath, time.strftime('%Y%m%d', time.localtime()))
    logfile = os.path.join(outpath,  'task_%s.log'%time.strftime('%Y%m%d', time.localtime()))

    return logfile

def save_mem_monitor(obj, savepic=1, logfile='', logpath='logs/', task=''):
    obj.addkey_mark('All finished')
    obj.stop()
    if logfile=='' and logpath !='':
        outpath = os.path.join(logpath, time.strftime('%Y%m%d', time.localtime()))
        logfile = os.path.join(outpath,  'task_%s.log'%time.strftime('%Y%m%d', time.localtime()))
    else:
        outpath = os.path.split(logfile)[0]
        os.makedirs(outpath, exist_ok=True)
    picpath = os.path.join(outpath, 'pic/')
    os.makedirs(picpath, exist_ok=True)
    if savepic == 1:
        picname = obj.mem_plotline(outpath=picpath, legend=0)
    keyline = copy.deepcopy(obj.keyline)

    txtout = []
    tl = 'keyline = %s' % ('\n'.join(map(str, keyline)) )
    txtout.append(tl)
    if len(keyline) == 8:
        score = single_model_score(keyline)
        tl = 'Single model metrics:'
        txtout.append(tl)
        tl = '%s' % ( '\t'.join(map(str, score.keys())) )
        txtout.append(tl)
        tl = '%s' % ( '\t'.join(map(lambda x:'%.3f'%x, score.values())) )
        txtout.append(tl)


    def calc_switch(data):
        
        tdat = [t for t,k,m,g in data if k in ['change_model_base', 'end_model_change']]
        ts = np.array(tdat[::2])
        te = np.array(tdat[1::2])
        sw_count = len(ts)
        sw_time = sum(te-ts) * 1000
        
        return sw_count, sw_time

    if task in ['worker_switch', 'test_worker_fcfs']:
        gdat = {}
        for t,k,m,g in keyline:
            if g=='0': continue
            
            if g in gdat.keys():
                gdat[g].append( (t,k,m,g) )
            else:
                gdat[g] = [(t,k,m,g)]
        sw_dat = {}
        for g, data in gdat.items():
            tmp = calc_switch(data)
            sw_dat[g] = tmp
            speed = 0 if tmp[0]==0 else tmp[1] / tmp[0]
            tl = 'GroupID:[%s] Switching times:%d, Total:%.3f(ms), Average:%.3f(ms)' % (g, tmp[0], tmp[1], speed)
            txtout.append(tl)
            print(tl)
        sw_count, sw_time = np.sum(list(sw_dat.values()), axis=0)
        tl = 'Total number of model switches:%d, total time:%.3f(毫秒)' % (sw_count, sw_time)
        txtout.append(tl)
        print(tl)
    tl = 'Maximum memory used:%.3fMB' % obj.mem_max()
    txtout.append(tl)
    print(tl)
    if picname:
        tl = 'VRAM image change graph:%s' % picname + '\n'
    else:
        tl= 'Failed to generate video memory image change map!\n'
    
    txtout.append(tl)
    print(tl)
        
    txt = '-'*40 + '\n' + '\n'.join(txtout) + '\n'
    if logfile:
        savetofile(txt, logfile, method='a+')
def rand_filename(path='', pre='', ext=''):
    nowtime = time.time()
    fmttxt = time.strftime('%Y%m%d%H%M%S', time.localtime(nowtime))
    filename = '%s%s%03d%s' % (pre, fmttxt, int((nowtime - int(nowtime))*1000), ext)
    fname = os.path.join(path, filename)
    return fname
def savetofile(txt, filename, encoding='utf-8', method='w'):
    pass
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt))
        return 1
    except :
        return 0

def load_data_lcqmc(filename, data_dir='', dat_length=0):
    D = []
    with open(data_dir + filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            text_a, text_b, label = line[0], line[1], line[-1]
            D.append((text_a, text_b, label))
            total_labels.add(label)
    if dat_length>0:
        D = D[:dat_length]
    return D

class MyDataset_lcqmc(DataGenerator):
    def __init__(self, examples, tokenizer=None, max_len=128):
        super(MyDataset_lcqmc, self).__init__(examples)
        self.total_labels = ['0','1']
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        examp = self.examples[index]
        text_a, text_b, label = examp[0], examp[1], examp[2]
        label_id = int(label)

        inputs = self.tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=self.max_len)
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        input_ids_tensor = sequence_padding(input_ids, self.max_len)
        att_mask_tensor = sequence_padding(attention_mask, self.max_len)
        segment_ids_tensor = sequence_padding(segment_ids, self.max_len)

        return {'token_ids': input_ids_tensor, 'attention_mask': att_mask_tensor, 'segment_ids': segment_ids_tensor, 'label': label_id}
def load_data_tnews(filename, data_dir='', dat_length=0):
    D = []
    with open(data_dir + filename, encoding='utf8') as f:
        for line in f:
            line = json.loads(line)
            try:
                sent = line['sentence']
                label = line.get('label', '100')
            except KeyError:
                continue
            D.append((sent, label))
            total_labels.add(label)
    if dat_length>0:
        D = D[:dat_length]
    return D

class MyDataset_tnews(DataGenerator):
    def __init__(self, examples, tokenizer=None, max_len=128):
        super(MyDataset_tnews, self).__init__(examples)
        self.total_labels = ['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116']
        self.label_map = {v:k for k,v in enumerate(self.total_labels)}
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        examp = self.examples[index]
        text, label = examp[0], examp[1]
        label_id = self.label_map.get(label, 0)

        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len)
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)

        input_ids_tensor = sequence_padding(input_ids, self.max_len)
        att_mask_tensor = sequence_padding(attention_mask, self.max_len)
        return {'token_ids': input_ids_tensor, 'attention_mask': att_mask_tensor, 'label': label_id}

def load_model_split(model_outpath, obj, model_base=None, model_task=None):
        print('loading model:%s...' % model_outpath)
    if obj: obj.addkey('start_base')
    model_base_file = os.path.join(model_outpath, 'model_base.pth')
    model_base = torch.load(model_base_file)
    if obj: obj.addkey('end_base')
    model_task_file = os.path.join(model_outpath, 'model_task.pth')
    if obj: obj.addkey('start_task_pb')
    model_task = torch.load(model_task_file)
    if obj: obj.addkey('end_task_pb')
    
    
    return model_base, model_task

def test_load_model_split():
    start = time.time()
    
    model_outpath = '../outputs/lcqmc/'
    model_base, model_task = load_model_split(model_outpath, None)
    print(model_base)
    print('model_base:', type(model_base))
    pl()
    print(model_task)
    print('model_task:', type(model_task))

    etime = time.time() - start
    print('etime:', etime)


def predict_task(model_base, model_task, 
                data_generator, 
                obj=None):
    preds = None

        return preds
    

if __name__ == '__main__':
    pass