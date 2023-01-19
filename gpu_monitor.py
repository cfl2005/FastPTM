#!/usr/bin/env python3
#coding:utf-8




import os
import sys
import time
import json
import numpy as np
from pynvml import *
import multiprocessing as MP
from datetime import datetime
def savetofile(txt, filename, encoding='utf-8', method='w'):
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt))
        return 1
    except Exception as e:
        return 0

def rand_filename():
    
    nowtime = time.time()
    fmttxt = time.strftime('%Y%m%d%H%M%S', time.localtime(nowtime))
    filename = '%s%03d' % (fmttxt, int((nowtime - int(nowtime))*1000) )
    return filename

def GPU_memory(gpuid:int=0):
    
    NUM_EXPAND = 1024 * 1024
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpuid)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used = info.used / NUM_EXPAND
    
    except Exception as e:
        gpu_memory_used = 0
    
    return  gpu_memory_used


def plotline(data, keydat=[], picname='./gpu_memory.png', legend=1):
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    if data == []: return 0
    
    try:
        fig = plt.figure(figsize=(10,5))
        dat = np.array(data)
        txt_date = datetime.fromtimestamp(dat[0,0]).strftime("%Y-%m-%d")
        x = [datetime.fromtimestamp(x) for x in dat[:, 0]]
        y = dat[:, 1]
        ymin, ymax = min(y)-10, max(y)+10
        plt.plot_date(x, y, fmt='bo', linestyle='-')
        int2color = lambda x: '#' + ('000000' + hex(int(x))[2:])[-6:]
        rand_color = lambda : int2color(time.time())
        n_rnd_color = lambda x:map(int2color, np.arange(65536, 16777216, 16711680//x))
        gids = sorted(set([x[3] for x in keydat]))
        colors = dict(zip(gids, n_rnd_color(len(gids))))
        if len(keydat)>0:
            i=0
            for t, n, v, g in keydat:
                i+=1
                col = rand_color()
                col = colors[g]
                txt = '%s:%.3f MB' % (n, v)
                xy = (datetime.fromtimestamp(t), v)
                plt.plot_date([xy[0], xy[0]], [ymin, ymax], 
                            linestyle='--', fmt='b', color=col, label=txt)
            if legend==1:
                plt.legend()
        plt.title('GPU Memory Report')
        plt.ylabel('GPU Memory(Mbyte)')
        plt.xlabel('Time(Seconds)')
        plt.grid()
        
        plt.savefig(picname)
        plt.close()
        return 1

    except Exception as e:
        print('Error in plotline: ')
        print(e)
        return 0
class GPU_MEM():
    def __init__(self, gpuid:int=-1, interval=1):

        if gpuid < 0:
            gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)

        self.gpuid = int(gpuid)     

        self.interval = interval
        self.status = 0
        manager = MP.Manager()
        self.mem_data = manager.list()
        self.keyline = manager.list()
        self.process = None
   
    def get_memory(self):
                mem = GPU_memory(self.gpuid)
        return mem

    def get_gpu_memory(self):
                while True:
            mem = self.get_memory()
            tm = time.time()
            dat = (tm, mem)
            self.mem_data.append(dat)

            if self.interval > 0:
                time.sleep(self.interval)
        
    def build(self, clear=1):
        if clear==1:
            manager = MP.Manager()
            self.mem_data = manager.list()
            self.keyline = manager.list()
        else:
            pass
        self.process = MP.Process(target=self.get_gpu_memory)#, args=(self.mem_data,))

    def start(self, interval=None, clear=1):
                try:
            if interval:
                self.interval = interval
            
            if self.process is None:
                self.build(clear=clear)

            self.process.start()
            self.status = 1
        except Exception as e:
            print(e)
            self.status = 0
        
    def stop(self):
                try:
            self.status = 0
            if not self.process is None:
                self.process.terminate()
                self.process = None
        except Exception as e:
            pass

    def restart(self):
                self.start(clear=0)

    def addkey_mark(self, keyname:str, groupid:str='0'):
        try:
            self.addkey(keyname, groupid=groupid)
        except :
            pass

    def addkey(self, keyname:str, delay=0, groupid:str='0'):
        if delay > 0: time.sleep(delay)
        
        tm = time.time()
        mem = GPU_memory(self.gpuid)
        dat = (tm, keyname, mem, groupid)
        self.keyline.append(dat)
        
        return dat

    def mem_ave(self):
                if self.mem_data == []:
            ret = 0
        else:
            dat = [v for t,v in self.mem_data]
            ret = np.average(dat)
        
        return ret
    
    def mem_max(self):
                if self.mem_data == []:
            ret = 0
        else:
            dat = [v for t, v in self.mem_data]
            ret = np.max(dat)
        return ret

    @staticmethod
    def abs_filter(data, n=10, delt=30):
                dat = data.copy()
        l = 0
        x0 = dat[0]
        ret = []
        for x in dat[1:]:
            if abs(x-x0) < delt:
                l+=1
                x = x0
            else:
                l=0
                
            if l>=n:
                if x not in ret and x!=0:
                    ret.append(x)
            x0 = x
        return ret
    
    def mem_segment(self, n=10, delt=20):
                if self.mem_data == []:
            ret = [0]
        else:
            dat = [v for t,v in self.mem_data]
            ret = self.abs_filter(dat, n=n, delt=delt)
       
        return ret

    def mem_plotline(self, picname='', outpath='./pic/', legend=1):
                if outpath == '': outpath = './'
        if picname=='':
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            picname = os.path.join(outpath, 'gpu_memory_%s.png' % rand_filename())

        ret = plotline(self.mem_data, keydat=self.keyline, picname=picname, legend=legend)
        if ret==0:
            picname = ''
        return picname

    def save(self, logfile):
        
        spath = os.path.split(logfile)[0]
        if not os.path.exists(spath):
            os.makedirs(spath)
        dats = {'mem_data': self.mem_data, 'keyline': self.keyline}
        dats_text = json.dumps(dats)
        savetofile(dats_text, logfile)

        return 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GPU监测')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID号')
    parser.add_argument('--seconds', type=int, default=30, help='持续时间')
    parser.add_argument('--interval', type=float, default=0.2, help='interval')

    args = parser.parse_args()
    gpu = args.gpu
    interval = args.interval
    seconds = args.seconds
    obj = GPU_MEM(gpuid=gpu, interval=interval)
    mem = obj.get_memory()
    print('Memory: %.4fMB' % mem)
    print('正在启动监控...')
    obj.start()
    time.sleep(1)
    obj.addkey('pause point')
    time.sleep(2)
    obj.addkey('pause point 2', groupid='group 2')
    print('外部访问数据:obj.mem_data:', obj.mem_data)
    time.sleep(seconds)
    obj.stop()
    print('-'*40)
    print('mem data:', obj.mem_data)
    print('mem keyline:', obj.keyline)
    print('mem_ave:', obj.mem_ave())
    print('mem_max:', obj.mem_max())
    print('mem_segment:', obj.mem_segment())
    picname = obj.mem_plotline()
    
    print('生成的曲线图文件名:%s' % picname)
