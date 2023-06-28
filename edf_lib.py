#!/usr/bin/env python3
#coding:utf-8

class GetCloseTime:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def greatest_common_divisor(self, _left, _right):
        return _left if _right == 0 else self.greatest_common_divisor(_right, _left % _right)

    def lowest_common_multiple(self):
        temp_result = 1
        for value in self.dictionary.values():
            temp_result = value[1] * temp_result / self.greatest_common_divisor(value[1], temp_result)
        return temp_result


class TaskControlBlock():
    CURRENT_TIME = 0

    def __init__(self, dictionary,
              name_list,
              period_time,
              central_processing_unit_time,
              remain_time,
              current_period):

        for key in dictionary.keys():
            name_list.append(key)
            period_time.append(dictionary.get(key)[1])
            central_processing_unit_time.append(dictionary.get(key)[0])
            remain_time.append(dictionary.get(key)[0])
            current_period.append(1)
        
        self.result = []


    def get_index_of_min(self, earliest_deadline_task_list, minimum):
        return earliest_deadline_task_list.index(minimum)

    def get_another_index_of_min(self, earliest_deadline_task_list, minimum):
        earliest_deadline_task_list[earliest_deadline_task_list.index(minimum)] = 100000
        return earliest_deadline_task_list.index(min(earliest_deadline_task_list))

    def is_execute(self, central_processing_unit_time, period_time):
        temp_list = [a / b for a, b in zip(central_processing_unit_time, period_time)]
        return sum(temp_list)

    def scheduling(self, name_list,
                period_time,
                central_processing_unit_time,
                remain_time,
                current_period):

        if self.is_execute(central_processing_unit_time, period_time) > 1:
            print("error, scheduling finish!")
            exit(0)
        
        earliest_deadline_task = self.get_index_of_min(
            [a * b for a, b in zip(period_time, current_period)],
            min(a * b for a, b in zip(period_time, current_period)))

        if self.CURRENT_TIME < period_time[earliest_deadline_task] * (current_period[earliest_deadline_task] - 1):
            current_period_p = period_time[earliest_deadline_task] * (current_period[earliest_deadline_task] - 1)
            temp_list = [a * b for a, b in zip(period_time, current_period)]
            while self.CURRENT_TIME < period_time[earliest_deadline_task] * \
                 (current_period[earliest_deadline_task] - 1):
                earliest_deadline_task = self.get_another_index_of_min(temp_list, min(temp_list))

            if remain_time[earliest_deadline_task] <= current_period_p - self.CURRENT_TIME:
                 running_time = remain_time[earliest_deadline_task]
            else:
                 running_time = current_period_p - self.CURRENT_TIME
            
            # current_period_p = period_time[earliest_deadline_task] * current_period[earliest_deadline_task]
            remain_time[earliest_deadline_task] -= running_time
            # 输出
            print(name_list[earliest_deadline_task], self.CURRENT_TIME, running_time)
            
            self.CURRENT_TIME += running_time
            if remain_time[earliest_deadline_task] == 0:
                current_period[earliest_deadline_task] += 1
                remain_time[earliest_deadline_task] = central_processing_unit_time[earliest_deadline_task]
        else:
            current_period_p = period_time[earliest_deadline_task] * current_period[earliest_deadline_task]
            if remain_time[earliest_deadline_task] <= current_period_p - self.CURRENT_TIME:
                running_time = remain_time[earliest_deadline_task]
            else:
                running_time = current_period_p - self.CURRENT_TIME
            remain_time[earliest_deadline_task] -= running_time
            # 输出
            print(name_list[earliest_deadline_task], self.CURRENT_TIME, running_time)
            
            self.CURRENT_TIME += running_time
            if remain_time[earliest_deadline_task] == 0:
                current_period[earliest_deadline_task] += 1
                remain_time[earliest_deadline_task] = central_processing_unit_time[earliest_deadline_task]


def EDF_sort(task_list):
    close_time_object = GetCloseTime(task_list)
    close_time = close_time_object.lowest_common_multiple()

    current_time = 0
    name_list = []
    period_time = []
    central_processing_unit_time = []
    remain_time = []
    current_period = []
    tcb = TaskControlBlock(task_list,
                        name_list,
                        period_time,
                        central_processing_unit_time,
                        remain_time,
                        current_period)
    
    while tcb.CURRENT_TIME < close_time:
        tcb.scheduling(name_list,
                    period_time,
                    central_processing_unit_time,
                    remain_time,
                    current_period)

    ret_idx = [otime.index(x.input_time) for x in ret]
    return ret_idx

def test_EDF_sort():
    import time
    task = [(0, 20),
            (4, 90),
            (50, 10),
            (8, 80),
            (7, 70),
            (20, 100),
        ]
    task_dictionary = {
                        "A": [10, 30],
                        "B": [20, 60],
                        "C": [30, 90]
                      }

    
    '''
    ntime = time.time() #int(time.time()*1000)
    print('ntime:', ntime)
    task = [(a+ntime,b) for a,b in task]
    '''

    ret = EDF_sort(task_dictionary)
    print('ret:', ret)


if __name__ == "__main__":

    test_EDF_sort()
