#!/usr/bin/env python3
#coding:utf-8




import argparse
from sgt_lib import *
import random
mp.set_start_method('spawn', force=True)

def max_process(num=1, logfile=''):
    print('"Maximum load test" in progress...')
    gpu_monitor = create_monitor()
    gpu_monitor.start()
    gpu_monitor.addkey_mark('test begin')
    args_lcqmc = conf_lcqmc()
    args_tnews = conf_tnews()
    args_os10 = conf_os10()

    args_lst = [args_lcqmc, args_tnews, args_lcqmc]
    total = len(args_lst)

    models = []
    for i in range(num):
        print('Creating model %d...'% (i+1))
        arg_idx = i % total
        task = Task_Model(tokenizer, None, obj=gpu_monitor)
        task.args = args_lst[arg_idx]
        task.load_model()
        models.append(task)
        time.sleep(0.2)
        
    for i in range(num):
        print('Calling model %d...'% (i+1))
        task = models[i]
        task.load_data()
        preds = task.predict()
        time.sleep(0.2)
    save_mem_monitor(gpu_monitor, savepic=1, logfile=logfile)


def max_worker(num=1, logfile=''):

    print('"Maximum WORKER load test" in progress...')
    gpu_monitor = create_monitor()
    gpu_monitor.start()
    gpu_monitor.addkey_mark('test begin')

    work = WORKER(gpu_monitor, isfull=0, preload=1)
    work.build(num)
    work.start()
    print('start sending data...')
    gpu_monitor.addkey('task_send')
    q = work.inqueue
    task_list = 'BC'
    task_list = 'BCABCABCABCA'
    task_list = 'ABC'

    total = len(task_list)

    print('Total number of tasks:%d'% total)
    for x in task_list:
        q.put(x)
        time.sleep(0.1)
    print('开始接收数据...')
    ret = []
    while 1:
        t = work.result.get()
        ret.append(t)
        if len(ret) >= total: break;
    
    print('received result:', len(ret))
    work.stop()
    save_mem_monitor(gpu_monitor, savepic=1, logfile=logfile)
def task_process(task_list, logfile=''):
    args_lcqmc = conf_lcqmc()
    args_tnews = conf_tnews()
    args_os10 = conf_os10()
    gpu_monitor = create_monitor()

    for task_code in task_list:
        task = Task_Model(tokenizer, None, obj=gpu_monitor)
        gpu_monitor.start()
        gpu_monitor.addkey_mark('test begin')
        task_code = task_code.upper()
                task.args = class_model_dict[task_code]
        
        task.args.task_type = task_code
        task.load_data(data_length=1000)
        task.load_model()
        print('model inference in progress...')
        preds = task.predict()
        print('preds:', len(preds))
        save_mem_monitor(gpu_monitor, savepic=1, logfile=logfile)
def test_switch(obj, isfull=0):
    args_lcqmc = conf_lcqmc()
    args_tnews = conf_tnews()

    if isfull==1:
        print('Complete Model Switching Test...')
    else:
        print('Split Model Switching Trials...')
    
    task = Task_Model(tokenizer, args_lcqmc, obj=obj)
    task.load_data(data_length=16)
    print('test_dataset:', task.test_dataset)
    if isfull==1:
        task.load_model_full()
    else:
        task.load_model()

    print('model inference in progress...')
    preds = task.predict()
    print('preds:', len(preds))
    print('Switching models...')
    task.args = args_tnews
    task.load_data(data_length=16)
    
    if isfull==1:
        task.load_model_full()
    else:
        task.load_model()

    print('model inference in progress...')
    preds = task.predict()
    print('preds:', len(preds))

def do_test_switch(isfull=0, logfile=''):
    gpu_monitor = GPU_MEM(interval=0.2)
    mem = gpu_monitor.get_memory()
    print('Memory: %.4fMB' % mem)
    gpu_monitor.build()
    gpu_monitor.start()

    test_switch(gpu_monitor, isfull=isfull)
    save_mem_monitor(gpu_monitor, savepic=1, logfile=logfile)
def save_task_detail(ret, savefile=''):
    txts = []
    tline = '-'*40
    txts.append(tline)
    rundat = []
    task_delay = [[], []]
    tline  = 'start,task,type,create,due,delay,timeout,end,group'
    rundat.append(tline)
    
    for ttt_dat, pdat, ftime, group_id in ret:
        ttt_dat_n = ttt_dat + (ftime, group_id)
        tline = ','.join(map(str, ttt_dat_n))
        rundat.append(tline)
        delay = round(ttt_dat[-2], 3)
        if ttt_dat[-1]:
            task_delay[0].append(delay)
        else:
            task_delay[1].append(delay)
    if savefile:
        tdat = '\n'.join(rundat)
        savetofile(tdat, savefile)
        print('The task run log file has been saved:%s' % savefile)

    count_delay = len(task_delay[0])
    tline = 'Number of overtime tasks:%d' % (count_delay)
    txts.append(tline)

    if count_delay>0:
        tline = 'delay:%s' % (task_delay[0][:20])
        txts.append(tline)

    count_ontime = len(task_delay[1])
    tline = 'Number of on-time tasks:%d' % (count_ontime)
    txts.append(tline)
    tline = 'delay:%s' % (task_delay[1][:20])
    txts.append(tline)

    txt = '\n'.join(txts)
    print(txt)
    return txt

def worker_switch(machines=1, groups=20, logfile='', 
                    isfull=0, task_list='', 
                    model_max_types=gbl_model_counts):

    print('Model switching test in progress"...')
    gpu_monitor = create_monitor()
    gpu_monitor.start()
    preload = 0
    work = WORKER(gpu_monitor, isfull=isfull, 
                preload=preload,model_max_types=model_max_types)

    work.build(machines)
    work.start()
    print('start sending data...')
    data_length = 16
    randtask = RandTask(total_types=model_max_types, data_length=data_length)

    if task_list == '':
        task_list = gbl_model_chars * groups

    total = len(task_list)
    gpu_monitor.addkey_mark('test begin')
    stime = time.time()
    print('Total number of tasks:%d'% total)
    for task in randtask.list_task(task_list):
        work.inqueue.put(task)
        time.sleep(0.1)
    print('start receiving data...')
    ret = []
    while 1:
        t = work.result.get()
        ret.append(t)
        if len(ret) >= total: break;
    
    print('received result:', len(ret))

    etime = time.time()
    utime = (etime - stime) * 1000
    print('Total prediction time: %.3f ms'%utime )
    work.stop()
    datfname = os.path.join(os.path.split(logfile)[0], 
                        '%s_SWITCH_m%d_g%d_t%d_isfull%d.csv' % (rand_id(), machines, groups, total, isfull) )
    save_task_detail(ret, savefile=datfname)
    save_mem_monitor(gpu_monitor, savepic=1, logfile=logfile, task='worker_switch')
    pl('', '=')


def test_worker_fcfs(machines, groups=4, 
                    task_list='',
                    interval=0.1, 
                    send_type=0, 
                    model_max_types=gbl_model_counts,
                    taskfile=''
                    ):
    obj = create_monitor()
    obj.start()
    data_length = 16
    tq = TaskQueue(total_types=model_max_types,
                    interval=interval, 
                    data_length=data_length)
    work = WORKER_FCFS(obj, inQueue=tq.queue)
    work.build(machines)
    work.start()

    print('waiting for process to start...')
    time.sleep(5)
    if taskfile:
        print('reading task sequence file...')
        taskdata = json.loads(readtxt(taskfile))
        total = len(taskdata)
        print('task sequence length:%d' % total)
        if total == 0:
            sys.exit()
        print('waiting for process to start...')
        time.sleep(5)
        print('Start sending random tasks...')
        obj.addkey_mark('test begin')
        stime = time.time()
        tq.timer_task_data(taskdata)
        tq.stop()
    else:
        if send_type<1:
            interval_items = 1
        else:
            interval_items = send_type
        if task_list == '':
            task_list = gbl_model_chars * groups

        total = len(task_list)
        print('start sending data: very %d sample interval %.2f seconds:...'% (interval_items, interval) )
        obj.addkey_mark('test begin')
        stime = time.time()

        tq.timer_task(task_list, interval_items=interval_items)
        tq.stop()
    print('start receiving data...')
    ret = []
    while 1:
        t = work.result.get()
        ret.append(t)
        if len(ret) >= total: break;

    etime = time.time()
    utime = (etime-stime)*1000
    print('Total prediction time: %.3f (ms)'%utime )
    work.stop()

    print('Total number of tasks:%d'% total)
    datfname = os.path.join(os.path.split(logfile)[0], 
                        '%s_FCFS_m%d_n%d_t%d_i%d.csv' % (rand_id(), machines, num, total, send_type) )

    save_task_detail(ret, savefile=datfname)

    save_mem_monitor(obj, savepic=1, task='test_worker_fcfs1')
    pl('', '=')

def test_scheduler(machines, num=4, 
                    logfile='', 
                    task_list='',
                    interval=0.1, 
                    send_type=0, 
                    model_max_types=gbl_model_counts,
                    taskfile=''):
    obj = create_monitor()
    obj.start()
    data_length = 16
    tq = TaskQueue(total_types=model_max_types, interval=interval, data_length=data_length)
    ts = TaskScheduler(tq.queue, model_num=machines, monitor=obj, model_max_types=model_max_types)
    if taskfile:
        print('reading task sequence file...')
        taskdata = json.loads(readtxt(taskfile))
        total = len(taskdata)
        print('task sequence length:%d' % total)
        if total == 0:
            sys.exit()

        ts.start(total)
        print('waiting for process to start...')
        time.sleep(5)
        print('Start sending random tasks...')
        obj.addkey_mark('test begin')
        stime = time.time()
        tq.timer_task_data(taskdata)
        tq.stop()

    else:
        if task_list == '':
            task_list = gbl_model_chars * num
        
        total = len(task_list)
        ts.start(total)
        if send_type<1:
            interval_items = 1
        else:
            interval_items = send_type
        print('waiting for process to start...')
        time.sleep(5)
        print('start sending data: Every %d sample interval %.2f seconds:...'% (interval_items, interval) )
        obj.addkey_mark('test begin')
        stime = time.time()
        tq.timer_task(task_list, interval_items=interval_items)
        tq.stop()
    print('receiving data...')
    ret = []
    while 1:
        t = ts.predict_queue.get()
        ret.append(t)
        if len(ret) >= total: break;
    
    etime = time.time()
    utime = (etime-stime)*1000
    print('Total prediction time: %.3f (ms)'%utime )
    ts.stop()
   
    print('Total number of tasks:%d'% total)
    datfname = os.path.join(os.path.split(logfile)[0], 
                        '%s_OMLS_m%d_n%d_t%d_i%d.csv' % (
                        rand_id(), machines, num, total, send_type)
                        )
    save_task_detail(ret, savefile=datfname)

    save_mem_monitor(obj, savepic=1, logfile=logfile)
    pl('','=')

def gen_task_list(task_list, num):
        tlist = []
    tmp  = list(task_list)
    for i in range(num):
        random.shuffle(tmp)
        tlist.extend(tmp)
    task_list = ''.join(tlist)
    return task_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wokers TEST')
    parser.add_argument('--task', type=str, required=True, default="", help='task name')
    parser.add_argument('--machines', type=int, default=1, help='machines')
    parser.add_argument('--num', type=int, default=4, help='How many sets of data=4')
    parser.add_argument('--isfull', type=int, default=0, help='1=complete model 0=split model')
    parser.add_argument('--task_list', type=str, default="", help='task list')
    parser.add_argument('--logfile', type=str, default="", help='logfile')
    parser.add_argument('--rnd', type=int, default=0, help='Is it random')
    parser.add_argument('--model_max_types', type=int, default=gbl_model_counts, help='Maximum number of models')
    parser.add_argument('--interval', type=float, default=0.1, help='sending speed')
    parser.add_argument('--send_type', type=int, default=0, help='interval type')
    parser.add_argument('--taskfile', type=str, default="", help='task data file')

    args = parser.parse_args()
    task = args.task.lower()
    machines = args.machines
    num = args.num
    isfull = args.isfull
    rnd = args.rnd
    task_list = args.task_list
    logfile = args.logfile
    model_max_types = args.model_max_types
    interval = args.interval
    send_type = args.send_type
    taskfile = args.taskfile
    if logfile == '':
        logfile = create_log_path('logs/')
    
    save_cmdline(logfile)
    print('logfile:', logfile)
    if num > 0 and taskfile=='':
        if task_list == '':
            task_list = gbl_model_chars
        if rnd == 1:
            task_list = gen_task_list(task_list, num)
        elif rnd==0:
            task_list = task_list * num
        elif rnd==2:
            pass

    if task in ['fcfs_test', 'fcfs']:
        test_worker_fcfs(machines, num,
                        task_list=task_list,
                        interval=interval,
                        send_type=send_type,
                        model_max_types=model_max_types,
                        taskfile=taskfile,
                        )
        
    if task == 'test_full_switch':
        do_test_switch(isfull=1, logfile=logfile)

    if task == 'task_process':
        task_process(task_list, logfile=logfile)
    
    if task == 'max_process':
        max_process(num, logfile=logfile)
    
    if task == 'max_worker':
        max_worker(num, logfile=logfile)

    if task == 'worker_switch':
        worker_switch(machines=machines, groups=num, 
                    logfile=logfile, isfull=isfull,
                    task_list=task_list, model_max_types=model_max_types)
        
    if task in ['test_scheduler', 'omls']:
        test_scheduler(machines, num=num,
                        logfile=logfile,
                        task_list=task_list,
                        interval=interval,
                        send_type=send_type,
                        model_max_types=model_max_types,
                        taskfile=taskfile,
                        )
        







