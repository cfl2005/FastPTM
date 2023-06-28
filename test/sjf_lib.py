#!/usr/bin/env python3
#coding:utf-8

class Homework(object):
    def __init__(self, input_time, run_time):
        self.tid = ''
        self.input_time = input_time
        self.run_time = run_time
        self.start_time = 0
        self.end_time = 0
        self.turn_around = 0
        self.dai_time = 0
        self.state = 0
 
    def zhouzhuan(self):
        self.turn_around = self.end_time - self.input_time
        return self.turn_around
 
    def daiquan(self):
        self.dai_time = (self.end_time - self.input_time) / self.run_time
        return self.dai_time
 
    def output(self):
        print("%d\t\t\t%d\t\t\t%d\t\t\t%d\t\t\t%d\t\t\t    %f" % (self.input_time,
                self.run_time, self.start_time, self.end_time, self.turn_around, self.dai_time))
        return self.tid

def sort(Homework):
    for i in range(len(Homework) - 1):
        for j in (i + 1, len(Homework) - 1):
            if Homework[i].input_time > Homework[j].input_time:
                t = Homework[j]
                Homework[j] = Homework[i]
                Homework[i] = t


def find_small(Homework):
    index = 65535
    rem = -1
    for i in range(len(Homework)):
        if Homework[i].state == 1 and Homework[i].run_time <= index:
            index = Homework[i].run_time
            rem = i
    return rem
 
def find_big(Homework, count):
    big_rate = -1
    index = -1
    for i in range(len(Homework)):
        rate = 1 + (count - Homework[i].input_time) / Homework[i].run_time
        if Homework[i].state == 1 and rate > big_rate:
            big_rate = rate
            index = i
    return index

def SJF(Homework):
    count = 0
    sort(Homework)
    # num = 0
    # weight_num = 0
    length = len(Homework)
    out = []
    while len(Homework):
        for j in Homework:
            if j.input_time > count:
                j.state = 0
            else:
                j.state = 1
        if find_small(Homework) == -1:
            Homework[0].state = 1
            count = Homework[0].input_time
        small = find_small(Homework)
        Homework[small].start_time = count
        count += Homework[small].run_time
        Homework[small].end_time = count
        Homework[small].turn_around = Homework[small].zhouzhuan()
        Homework[small].dai_time = Homework[small].daiquan()
        # Homework[small].output()
        out.append(Homework[small])
        # num += Homework[small].turn_around
        # weight_num += Homework[small].dai_time
 
        del Homework[small]
    
    return out
    '''
    average = num / length
    weight_average = weight_num / length

    print("T = %f" % average)
    print("F = %f" % weight_average)
    '''

def SFJ_sort(task_list):
    otime = [a for a,b in task_list]
    homework = [Homework(a, b) for a,b in task_list]
    ret = SJF(homework)
    ret_idx = [otime.index(x.input_time) for x in ret]
    return ret_idx

def test_SFJ_sort():
    import time
    task = [(0, 20),
            (4, 90),
            (50, 10),
            (8, 80),
            (7, 70),
            (20, 100),
        ]
    
    ntime = time.time() #int(time.time()*1000)
    print('ntime:', ntime)
    task = [(a+ntime,b) for a,b in task]
    
    '''
    homework1 = [h1, h2, h3, h4]
    homework2 = [h1, h3, h5, h6]
    homework3 = [h1, h2, h3, h4]
    homework4 = [h1, h3, h5, h6]
    homework5 = [h1, h2, h3, h4]
    homework6 = [h1, h3, h5, h6]
    '''

    ret = SFJ_sort(task[:5])
    print('ret:', ret)

def test_SJF():
    h1 = Homework(0, 20)
    h2 = Homework(4, 90)
    h3 = Homework(50, 10)
    h4 = Homework(8, 80)
    h5 = Homework(7, 70)
    h6 = Homework(20, 100)

    homework1 = [h1, h2, h3, h4]
    homework2 = [h1, h3, h5, h6]
    homework3 = [h1, h2, h3, h4]
    homework4 = [h1, h3, h5, h6]
    homework5 = [h1, h2, h3, h4]
    homework6 = [h1, h3, h5, h6]

    ret = SJF(homework3)
    print(type(ret[0]))
    print('-'*40)
    print("SJF:")
    print("input_time\trun_time\tstart_time\tend_time\tturnover_time\tweighted_turnover_time\t")
    num = 0
    weight_num = 0
    length = len(ret)

    for x in ret:
        num += x.turn_around
        weight_num += x.dai_time
        x.output()
    
    average = num / length
    weight_average = weight_num / length

    print("T = %f" % average)
    print("F = %f" % weight_average)


if __name__ == '__main__':
    pass
    # test_SJF()
    test_SFJ_sort()
