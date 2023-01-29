#!/usr/bin/env python3
#coding:utf-8



import argparse
import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd

pl = lambda x='', y='-': print(y*40) if x=='' else print(x.center(40, y))
pr = lambda x: print('%s:%s' % (x, eval(x)))
def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''
def savetofile(txt, filename, encoding='utf-8', method='w'):
    pass
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt))
        return 1
    except Exception as e:
        return 0


def findkey(txt, rules):
        ret = {}
    for k, v in rules:
        pat = re.compile(v)
        sret = re.findall(pat, txt)
        if sret:
            ret[k] = ', '.join(sret)
    
    return ret

logfile = 'nlogs/test_task_mu01_202301051118.log'
logfile = 'nlogs/test_task_mu01_202301051247.log'
logfile = 'nlogs/test_task_mu005_202301060844.log'
logfile = 'nlogs/test_task_mu008_202301060916.log'


parser = argparse.ArgumentParser(description='log data to xls')
parser.add_argument('--task', type=str, required=True, default="", help='task')
parser.add_argument('--logfile', type=str, default="", help='log file')

args = parser.parse_args()
task = args.task
logfile = args.logfile
rules = [
    ('cmd',             r'>> (python(?:.+))\n'),
    ('task',            r'--task=(\w+) '),
    ('machines',        r'--machine=(\d+) '),
    ('isfull',          r'--isfull=(\d+)'),
    ('num',             r'--num=(\d+) '),
    ('groups',          r'_T(\d+)_'),
    ('models',          r'_L(\d+)_'),
    ('total_time',      r'Forecast total time:([\d\.]+)ms'),
    ('total_task',      r'Total number of tasks:(\d+)'),
    ('over_task',       r'Number of overtime tasks:(\d+)'),
    ('intime_task',     r'Number of on-time tasks:(\d+)'),
    ('change_times',    r'Total:([\d\.]+)\(ms\)'),
    ('change_AVE',      r'Average:([\d\.]+)\(ms\)'),
    ('changelist',      r'Switching times:(\d+)'),
    ('changelist',      r'Model switching times:\[([\d, ]+)\]'),
    ('total_change',    r'total switching times:(\d+)'),
    ('total_change',    r'Total number of model switches:(\d+)'),
    ('memory',          r'Maximum memory used:([\d\.]+)MB'),
    ('memory_image',    r'VRAM image change graph:((?:[\w/\.]+)png)'),
]


def task_mu_to_xls(logfile):
    logtxt = readtxt(logfile)
    if logtxt == '':
        print('Unable to read log file...')
        sys.exit()

    kline = '='*40

    datlist = logtxt.split(kline)
    total = len(datlist)

    print('There are %d records in total.' % total)

    results = []
    for txt in datlist[:]:
        dat_dict = findkey(txt, rules) 
        if dat_dict:
            results.append(dat_dict)
        data_sheet = []
    for item in results:
        if not item: continue
        
        scence = '012345678'[ int(item['machines'])] + ' machiine(s)'
        models = int(item['models'])
        mod_dat = '%d Models' % models
        t = int(item['groups'])/models
        gp = '%d Groups ' % t
        th = item['task'].upper()
        mem = item['memory'] 
        change = 'switch:%s total switch:%s' % (item['changelist'], item['total_change'] )
        over = item['over_task']
        total = item['total_time']
        dline = [scence, mod_dat, gp, th, mem, change, over, total]
        data_sheet.append(dline)

    columns = ['Testing Scenes','Number of models', 'Test case', 
                'Algorithm', 'Memory usage (MB)',
                'Switching times','Number of overtime tasks',
                'Total task time']
    df = pd.DataFrame(data_sheet, columns=columns)
    pl()
    print(df.head())

    outfile = logfile.replace('.log', '.xls')
    df.to_excel(outfile)

def task_switch_to_xls(logfile):
    logtxt = readtxt(logfile)
    if logtxt == '':
        print('Unable to read log file...')
        sys.exit()

    kline = '='*40

    datlist = logtxt.split(kline)

    results = []
    for txt in datlist[:]:
        dat_dict = findkey(txt, rules) 
        if dat_dict:
            print('dat_dict:', json.dumps(dat_dict,indent=4))
            pl()
            results.append(dat_dict)

    total = len(results)
    print('共有记录%d条。' % total)
    print('results:', results)
    data_sheet = []
    for item in results:
        if not item: continue
        
        scence = '012345678'[ int(item['machines'])] + 'machines'
        isfull = int(item.get('isfull', 0))
        isfulltxt = ['weight switch', 'model switching'][isfull]
        num = (item['num'])                                     
        nums = '%s Groups' % num
        mem = item['memory']
        changelist = item['changelist'].split(', ')
        change_times = item['change_times'].split(', ')
        change_AVE = item['change_AVE'].split(', ')
        print('changelist:', changelist)
        print('change_times:', change_times)
        print('change_AVE:', change_AVE)
        change_dat = zip(changelist, change_times, change_AVE)
        
        change_txt = '\n'.join( map(str, change_dat))
        total = item['total_time']
        dline = [scence, isfulltxt, nums, mem, change_txt, total]
        data_sheet.append(dline)

    columns = ['Scenes', 'Test object', 'Test case', 
                'Memory usage (MB)','Switching times', 'Total task time'
              ]
    df = pd.DataFrame(data_sheet, columns=columns)
    print(df.info())
    pl()
    print(df.head())

    outfile = logfile.replace('.log', '.xls')
    df.to_excel(outfile)

if __name__ == '__main__':
   
    if task == 'mu':
        if os.path.isfile(logfile):
            task_mu_to_xls(logfile)
        elif os.path.isdir(logfile):
            print('processed in batches...')
            workpath = logfile
            for dirname in os.listdir(workpath):
                fext = os.path.splitext(dirname)[1]
                if fext in ['.log']:
                    file_path = os.path.join(workpath, dirname)
                    print('Processing file::%s'%file_path)
                    task_mu_to_xls(file_path)

    if task == 'switch':
        task_switch_to_xls(logfile)


