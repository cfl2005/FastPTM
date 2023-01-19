#!/usr/bin/env python3
#coding:utf-8



import os
import sys
import logging
import time
import torch.nn as nn
import torch.utils.checkpoint
sys.path.insert(0, '../')
from bert4pytorch.configs.configuration_bert import BertConfig
from bert4pytorch.tokenizers.tokenization_bert import BertTokenizer
from bert4pytorch.trainers.train_func import Trainer
from bert4pytorch.snippets import set_seed, sequence_padding, DataGenerator
from bert4pytorch.models.model_building import *
from bert4pytorch.trainers.logger_utils import logger
from tqdm import tqdm, trange
from multiprocessing import Value

from data_lib import *
from config import *

bert_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
tokenizer = BertTokenizer(bert_path + "/vocab.txt")
time_baseline = time.time()

pl = lambda x='', y='-': print(y*40) if x=='' else print(x.center(40, y))
def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''
def save_cmdline(logfile):
    cmd_line = ' '.join(sys.argv)
    tmtxt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    txt = '\n>>[%s]>> python %s \n' % (tmtxt, cmd_line)
    print(txt)
    savetofile(txt, logfile, method='a+')
def rand_id():
    nowtime = time.time()
    fmttxt = time.strftime('%Y%m%d%H%M%S', time.localtime(nowtime))
    filename = '%s%03d' % (fmttxt, int((nowtime - int(nowtime))*1000) )
    return filename
def rand_txt(maxlen=10):
    txt = ''.join([np.random.choice(list(string.ascii_letters)) 
        for i in range(np.random.randint(maxlen)+1)])
    return txt

class Task():
        etime = (time.time() - self.timestamp) * 1000
        speed = table_model_predict_speed[self.task_type]
        ti = self.total * speed
        li = table_model_loaded[self.task_type]
        pri = (self.deadline*1000 - etime) - ti - li
        ret = pri  - switch_threshold 
        return int(ret)

    def load_data(self):
        task = self.task_cfg
        dataset = task['dataset']
        mtype = task['mtype']
        data_path = task['data_path']
        label_dict_file = os.path.join(data_path, 'labels.txt')
        label_dict = load_label_dict(label_dict_file)
        label2id = {j: i for i, j in label_dict.items()}
        num_classes = len(label_dict)
        categories = list(label_dict.values())
        datlength = self.data_length
        batch_size = 32
        maxlen = 512
        if dataset in ['os10']:
            train_data, valid_data, test_data = load_data_os10(data_path, length=datlength)
        if dataset == 'tnews':
            train_data, valid_data, test_data = load_data_tnews(data_path, length=datlength, label_dict=label_dict)
        if dataset == 'lcqmc':
            train_data, valid_data, test_data = load_data_lcqmc(data_path, length=datlength)
        if dataset == 'people':
            train_data, valid_data, test_data = load_data_people(data_path, length=datlength)
        if mtype=='cls':
            test_generator = data_generator(
                            tokenizer, maxlen, num_classes,
                            test_data, batch_size=batch_size)
        
        if mtype=='ner':
            test_generator = data_generator_ner(
                            tokenizer, maxlen, label2id, 
                            test_data, batch_size=batch_size)
        self.data_generator = test_generator

class Task_Model():
    def __init__(self, tokenizer, args, obj=None, isfull=0):
        self.gid = str(int(time.time()*1000))

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obj = obj

        self.model_base = None
        self.model_task = None
        self.tokenizer = tokenizer

        self.test_dataset = None
        self.test_data_iterator = None

        self.model_path = ''
        self.change = Value('i', 0)
        self.isfull = isfull
        self.last_model = ''
    
    def change_times(self):
        value = self.change.value
        return value

    def load_data(self, data_length=16):

        if self.args.dataset_name == 'lcqmc':
            test_examples = load_data_lcqmc('test.txt', self.args.data_dir, dat_length=data_length)
            self.test_dataset = MyDataset_lcqmc(test_examples, 
                                        tokenizer=self.tokenizer, 
                                        max_len=self.args.max_len)
        
        if self.args.dataset_name == 'tnews':
            test_examples = load_data_tnews('dev.json', self.args.data_dir, dat_length=data_length)
            self.test_dataset = MyDataset_tnews(test_examples, 
                                        tokenizer=self.tokenizer, 
                                        max_len=self.args.max_len)

        self.test_data_iterator = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=1,
        )
    
    def addkey(self, keyname):
        if not self.obj is None:
            
            self.obj.addkey_mark(keyname, self.gid)

    def load_model(self, args=None):
        if args is None:
            args = self.args
        output_model_base = os.path.join(args.model_outpath, "model_base.pth")
        output_model_task = os.path.join(args.model_outpath, "model_task.pth")
        output_model_base_weight = os.path.join(args.model_outpath, "model_base.bin")
        output_model_task_weight = os.path.join(args.model_outpath, "model_task.bin")
        if (os.path.exists(output_model_base) and os.path.exists(output_model_task)):
            if self.last_model != args.task_type: 
                if self.model_base is None:
                    print('首次加载模型...')
                    self.addkey('load_model_base')
                    self.model_base = torch.load(output_model_base)
                    self.addkey('end_model_base')

                    self.addkey('load_model_task')
                    self.model_task = torch.load(output_model_task)
                    self.addkey('end_model_task')
                
                else:
                    self.addkey('change_model_base')
                    self.model_base.bert.load_state_dict(torch.load(output_model_base_weight))
                    v = self.change.value + 1
                    self.change.value = v
                    self.model_task = torch.load(output_model_task)
                    self.addkey('end_model_change')
                self.model_path = args.model_outpath
                self.last_model = args.task_type
        else:
            if self.model_base is None:
                kwargs = {"num_labels": args.num_labels}
                config = BertConfig.from_pretrained(args.model_path, **kwargs)
                self.model_base = BertBase(config=config, model_path=args.model_path, model_name='bert')

            self.model_task = model_task_tnews(config)
        
            print('正在加载模型权重...')
            self.model_base.bert.load_state_dict(torch.load(output_model_base_weight))
            self.model_task.load_state_dict(torch.load(output_model_task_weight))
            
    def load_model_full(self):
        args = self.args

        if self.model_base is None:
            kwargs = {"num_labels": args.num_labels}
            config = BertConfig.from_pretrained(args.model_path, **kwargs)
            if args.dataset_name == "lcqmc":
                model = Model_lcqmc(config=config, model_path=args.model_path, model_name=args.model_name)
            if args.dataset_name == "tnews":
                model = Model_tnews(config=config, model_path=args.model_path, model_name=args.model_name)
            self.model_base = model.bert
            self.model_task = model.task

            self.model_base.to(self.device)
            self.model_task.to(self.device)
            output_model_base_weight = os.path.join(args.model_outpath, "model_base.bin")
            output_model_task_weight = os.path.join(args.model_outpath, "model_task.bin")

            self.addkey('load_model_base')
            self.model_base.load_state_dict(torch.load(output_model_base_weight))
            self.addkey('end_model_base')

            self.addkey('load_model_task')
            self.model_task.load_state_dict(torch.load(output_model_task_weight))
            self.addkey('end_model_task')

        else:
            if self.last_model != args.task_type: 
                self.addkey('change_model_base')
                kwargs = {"num_labels": args.num_labels}
                config = BertConfig.from_pretrained(args.model_path, **kwargs)
                if args.dataset_name == "lcqmc":
                    model = Model_lcqmc(config=config, model_path=args.model_path, model_name=args.model_name)
                if args.dataset_name == "tnews":
                    model = Model_tnews(config=config, model_path=args.model_path, model_name=args.model_name)
                self.model_base = model.bert
                self.model_task = model.task

                self.model_base.to(self.device)
                self.model_task.to(self.device)
                output_model_base_weight = os.path.join(args.model_outpath, "model_base.bin")
                output_model_task_weight = os.path.join(args.model_outpath, "model_task.bin")

                self.model_base.load_state_dict(torch.load(output_model_base_weight))
                self.model_task.load_state_dict(torch.load(output_model_task_weight))

                self.addkey('end_model_change')

    def predict(self):
        self.model_base.to(self.device)
        self.model_task.to(self.device)

        self.model_base.eval()
        self.model_task.eval()
        self.addkey('start_predict')
        
        preds = None
        for batch in self.test_data_iterator:
            with torch.no_grad():
                batch_x = {}
                for name, item in batch.items():
                    if name != 'label':
                        batch_x[name] = item.to(self.device)

                outputs1 = self.model_base(**batch_x)
                if self.isfull==1:
                    outputs1 = outputs1[0]
                outputs = self.model_task(outputs1)
                logits = outputs
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        
        self.addkey('end_predict')
        return preds
class IModel():
    def __init__(self, outque=None, load_time=0, speed=0, monitor=None):
        if load_time==0:
            load_time = np.random.randint(2,10)
        self.load_time = load_time
        self.speed = speed
        self.queue = mp.Queue()
        self.outque = outque
        self.current_task_type = -1
        self.p = None
        self.w = None
        self.model_outpath = ''
        self.task_name = ''
        self.switch_times = Value('i', 0)
        self.model_base = None
        self.model_task = None
        self.obj = monitor
        self.c_status = Value('i', 0)
        self.task = Task_Model(tokenizer, None, obj=monitor)
    def status(self):
        v = self.c_status.value
        q = self.queue.qsize()
        s = int(not ((v==0) and q==0))
        return s
    def switchs(self):
        val = self.switch_times.value
        return val

    def put(self, dat):
        self.queue.put(dat)

    def do_work(self, dofun):
        while 1:
            task = self.queue.get()
            dofun(task)

    def prepair(self, task_type):
        task = Task(task_type)
        self.current_task_type = task.task_type
        ta = task.task_char
        self.task.args = class_model_dict[ta]
        self.task.args.task_type = ta
        self.task_name = ta
        self.task.load_model()

    def do_queue(self): 
                while 1:
            task = self.queue.get()
            self.c_status.value = 1
            model_path = task.task_cfg['model_path']
            self.current_task_type = task.task_type

            ta = task.task_char
            self.task.args = class_model_dict[ta]
            self.task.args.task_type = ta
            
            ttt_dat = task.get_key_dat()
            self.task.load_data(data_length=16)
            self.task.load_model()
            ret = self.task.predict()
            if self.outque:
                ftime = time.time()
                group_id = self.task.gid
                self.outque.put([ttt_dat, ret, ftime, group_id])
            if self.task_name == '':
                self.task_name = ta
            else:
                if self.task_name != ta:
                    svalue = self.switch_times.value + 1
                    self.switch_times.value = svalue
                    self.task_name = ta
            self.c_status.value = 0

    def start(self, task_type=-1):
        if task_type>0:
            self.prepair(task_type)
        self.p = mp.Process(target=self.do_queue)#, args=(self.queue, ))
        self.p.start()
    
    def stop(self):
        try:
            if not self.p is None:
                self.p.terminate()
        except Exception as e:
            pass

    def load_model(self, model_outpath):
        if self.model_outpath != model_outpath:
        
                        self.task.load_model()
            self.model_outpath = model_outpath
            self.switch_times += 1
    
    def predict(self, data_generator, datlength=1):
                return ret

    def load_model_1(self, model_outpath):
        if self.model_outpath != model_outpath:
            self.model_base, self.model_task = load_model_all(model_outpath, self.obj,
                                                model_base=None, 
                                                model_task=None)
            self.model_outpath = model_outpath
            self.switch_times += 1

    def predict_1(self, data_generator, datlength=1):
        ret = predict_task(self.model_base, self.model_task,
                            data_generator, datlength=datlength, 
                            obj=self.obj)
        return ret

class RandTask():
        def __init__(self, total_types=gbl_model_counts, data_length=10):
        pass
        self.total_types = total_types
        self.data_length = data_length

    def rand_task(self):
        task_type = np.random.randint (self.total_types)
        ntask = Task(task_type=task_type, data_length=self.data_length)
        return ntask

    def list_task(self, tasklist):
        for t in tasklist:
            idx = gbl_model_chars.index(t)
            ntask = Task(task_type=idx, data_length=self.data_length)
            yield ntask
        
class TaskQueue():
        def __init__(self, total_types=gbl_model_counts, task_count:int=gbl_model_counts, interval=0.1, data_length=16):
        self.interval = interval
        self.queue = Queue()
        self.stop_event = 0
        self.process = None
        self.total_types = total_types
        self.task_count = task_count
        
        self.randtask = RandTask(total_types=total_types, data_length=data_length)
    
    def timer_task(self, tasklist='', interval_items=1):
                items = 0
        if tasklist != '':
            for task in self.randtask.list_task(tasklist):
                self.queue.put(task)
                items += 1
                time.sleep(0.01)
                if items >= interval_items:
                    items = 0
                    time.sleep( self.interval )
        else:
            while 1:
                ntask = self.randtask.rand_task()
                self.queue.put(ntask)
                items += 1
                if items >= interval_items:
                    items = 0
                    time.sleep( self.interval )

    def timer_task_data(self, taskdata):
                for tasklist, interval in taskdata:
            taskg = self.randtask.list_task(tasklist)
            tasks = list(taskg)
            task = tasks[0]
            self.queue.put(task)
            time.sleep(interval)

    def start(self, tasklist):
        if self.process is None:
            self.process = Process(target=self.timer_task, args=(tasklist,)) #
        self.stop_event = 0
        self.process.start()
    
    def print_queue(self):
        while 1:
            q = self.queue.get()
            pri = q.get_priority()
            time.sleep(0.5)

    def start_print(self):
        self.p = Process(target=self.print_queue)
        self.p.start()

    def do_queue(self, dofun):
        while 1:
            q = self.queue.get()
            dofun(q)

    def stop(self):
        try:
            if not self.process is None:
                self.process.terminate()
            if not self.p is None:
                self.p.terminate()
        except Exception as e:
            pass
class TaskScheduler():
        def __init__(self, inQueue:mp.Queue, model_num:int=2, monitor=None, model_max_types=gbl_model_counts):
       
        self.starttime = time.time()
        self.inQueue = inQueue
        self.model_num = model_num
        self.model_max_types = model_max_types
        self.predict_queue = mp.Queue()
        self.outQueue = [mp.Queue() for i in range(self.model_max_types)]
        self.task_cache = []
        self.pri_que = mp.Queue()
        self.monitor = monitor
        self.models = [IModel(outque=self.predict_queue, monitor=monitor) for i in range(model_num)]

        self.process = None
        self.pusher = None
        self.resulter = None
        self.status = 0
        self.obj = monitor
        self.finished = 0
        self.start_model()

    def get_task(self):
        qsizes = np.array([x.queue.qsize() for x in self.models])
        idle_model = np.where(qsizes==0)[0]

    def process_push(self):
        while 1:
            task = self.inQueue.get()
            itype = task.task_type
            self.outQueue[itype].put(task)
    def process_scheduler(self):
        while 1:
            self.do_schedule()
            time.sleep(0.01)
    def result_scheduler(self, queue, total=0):
        finished = 0
        while 1:
            ret = queue.get()
            finished += 1
            print('已完成任务数:%d'%finished)
            if finished>=total:
                self.stop()
                self.finished = 1
                break;

    def start_model(self):
        print('正在启动推理模型...')
        for i, m in enumerate(self.models):
            m.start(task_type=i)
        
    def start(self, total):
        if self.process is None:
            self.process = mp.Process(target=self.process_scheduler)
        self.process.start()
        if self.pusher is None:
            self.pusher = mp.Process(target=self.process_push)
        self.pusher.start()

        
        self.status = 1

    def stop(self):
                try:
            print('正在结束进程...')
            if not self.pusher is None:
                self.pusher.terminate()
                self.pusher = None
            if not self.process is None:
                self.process.terminate()
                self.process = None
            
            self.status = 0
            all_switch_times_list = [x.switchs() for x in self.models]
            all_switch_times = sum(all_switch_times_list)
            print('模型切换次数:%s, 总切换次数:%d' % (all_switch_times_list, all_switch_times)) 

            for x in self.models:
                x.stop()

        except Exception as e:
            print(traceback.format_exc())
            print(e)
    def priority (self, tasks):
        task = d[0]
        ret = task.timestramp - self.starttime

    def do_schedule(self):
        len_cache = len(self.task_cache)
        list_task_type = []
        for x in self.models:
            c_type = x.current_task_type
            if c_type >= 0:
                list_task_type.append(c_type)
        if len_cache==0 and 1:

            for i in range(len(self.outQueue)):
                if i in list_task_type: continue
                queue = self.outQueue[i]

                if queue.qsize() > 0:
                    tmp_task = queue.get()
                    self.task_cache.append(tmp_task)
            len_cache = len(self.task_cache)
            if len_cache>0:
                pass

        
        len_cache = len(self.task_cache)
        pri_list = [x.get_priority() for x in self.task_cache]

        if len(pri_list) > 0:
            max_pri = np.argmin(pri_list)
            max_task_pri = pri_list[max_pri]
        else:
            max_pri = -1
        status = [x.status() for x in self.models]
        status = np.array(status)
        isIdle = len(np.where(status==0)[0])>0
        if not isIdle :
            return 1
        else:
            pass
        if len(pri_list) > 0 and max_task_pri < 0:#
            
            task = self.task_cache.pop(max_pri)
            itype = task.task_type
            idle_dat = np.where(status==0)[0]
            if len(idle_dat)>0:
                idle_idx = idle_dat[0]
                machine = self.models[idle_idx]
                machine.queue.put(task)
                machine.current_task_type = itype

        else:
            for i in range(len(self.models)):
                machine = self.models[i]
                st = machine.status()
                if st==1:
                    continue
                current_task_type = machine.current_task_type
                if current_task_type < 0 :
                    if len(self.task_cache)>0:
                        tmp_task = self.task_cache.pop(0)
                        tid = tmp_task.task_id
                        tchar = tmp_task.task_char
                        machine.current_task_type = tmp_task.task_type
                        machine.queue.put(tmp_task)

                else:
                    queue = self.outQueue[current_task_type]
                    if queue.qsize()>0:
                        tmp_task = queue.get()
                        tid = tmp_task.task_id
                        tchar = tmp_task.task_char
                        machine.queue.put(tmp_task)
                    else:
                        len_cache = len(self.task_cache)
                        if len_cache>0:
                            task = self.task_cache.pop(0)
                            itype = task.task_type
                            tid = task.task_id
                            tchar = task.task_char
                            machine.current_task_type = itype
                            machine.queue.put(task)

                        
                        
        return 1
class WORKER():
    def __init__(self, obj, isfull=0, preload=0, model_max_types=gbl_model_counts):
        self.obj = obj
        self.inqueue = mp.Queue()
        self.result = mp.Queue()

        self.isfull = isfull
        self.preload = preload

        self.process = []
        self.task = []
        self.machines = 0
        self.in_process = None

        self.model_max_types = model_max_types
        self.tque = [mp.Queue() for i in range(self.model_max_types)]
        self.groupid = str(int(time.time()*1000))

    def do_inque(self):
        idx = 0
        while 1:
            task = self.inqueue.get()
            ta = task.task_char
            idx = (idx+1) % self.machines
            outque = self.tque[idx].put(task)

    def do_worker(self, queue, outque):
        task = Task_Model(tokenizer, None, self.obj, isfull=self.isfull)
        if self.preload==1:
            task.args = conf_tnews()
            task.args.task_type = 'A'
            if self.isfull == 1:
                task.load_model_full()
            else:
                task.load_model()

        self.task.append(task)
        while 1:
            ntask = queue.get()
            ta = ntask.task_char

            task.args = class_model_dict[ta]
            task.args.task_type = ta
            task.load_data(data_length=16)
            if self.isfull == 1:
                task.load_model_full()
            else:
                task.load_model()
            ttt_dat = ntask.get_key_dat()
            preds = task.predict()
            ftime = time.time()
            outque.put([ttt_dat, preds, ftime, task.gid])

    def build(self, num):
        self.in_process = mp.Process(target=self.do_inque)
        self.machines = num
        self.tque = [mp.Queue() for i in range(num)]
        for i in range(num):
            que = self.tque[i]
            t = mp.Process(target=self.do_worker, args=(que, self.result))
            self.process.append(t)
            time.sleep(0.1)
        
    def start(self):
        self.in_process.start()
        for x in self.process:
            x.start()
        print('Process started.')

    def stop(self):
        try:
            self.in_process.terminate()
            for x in self.process:
                x.terminate()
            self.process = []
            
            
        except Exception as e:
            pass
class WORKER_FCFS():
    def __init__(self, obj, isfull=0, inQueue=None):
        self.obj = obj
        if inQueue is None:
            self.inqueue = mp.Queue()
        else:
            self.inqueue = inQueue
        self.result = mp.Queue()
        self.tasks = []
        self.process = []
        self.machines = 0

        self.data_length = 16

        self.in_process = None
        self.isfull = isfull
        self.groupid = str(int(time.time()*1000))

        self.status = 0
        self.models = []

    def do_trans(self, inqueue):
        idx = 0
        while 1:
            time.sleep(0.01)
            task = inqueue.get()

            machine = self.models[idx]
            machine.queue.put(task)
            idx = (idx + 1) % self.machines

    def do_worker(self, queue, outque):

        task = Task_Model(tokenizer, None, self.obj)
        self.tasks.append(task)
        while 1:
            time.sleep(0.01)
            ntask = queue.get()
            ta = ntask.task_char
            task.args = class_model_dict[ta]
            task.args.task_type = ta
            task.load_data(data_length=16)
            if self.isfull == 1:
                task.load_model_full()
            else:
                task.load_model()
            ttt_dat = ntask.get_key_dat()

            preds = task.predict()
            ftime = time.time()
            group_id = task.gid
            outque.put([ttt_dat, preds, ftime, group_id])

    def build(self, num):
                self.machines = num
        self.models = [IModel(outque=self.result, monitor=self.obj) for i in range(num)]

        self.process = mp.Process(target=self.do_trans, args=(self.inqueue,))

        
    def start(self):
        
        for i, m in enumerate(self.models):
            m.start(task_type=i)

        self.process.start()
        print('Processes are all started.')

    def stop(self):
        try:
            self.status = 0
            all_switch_times_list = [x.switchs() for x in self.models]
            all_switch_times = sum(all_switch_times_list)
            print('Model switching times:%s, total switching times:%d' % (all_switch_times_list, all_switch_times)) 

            for x in self.models:
                x.stop()
            
            self.process.terminate()
        except Exception as e:
            pass
    

if __name__ == '__main__':
    pass

