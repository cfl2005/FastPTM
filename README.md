# FastPTM : Fast Weight Loading for Inference Acceleration of Pre-Trained Models in GPUs
Pre-trained models (PTMs) have demonstrated great success in a variety of NLP and CV tasks and have become a significant development in the field of deep learning. However, the large memory and high computational requirements associated with PTMs can increase the cost and time of inference, limiting their usability in practical applications. To improve the user experience of PTM applications by reducing waiting and re- sponse times, we propose the FastPTM framework. This general framework aims to accelerate PTM inference tasks by reducing task loading time and task switching overhead while deployed on edge GPUs. The framework utilizes a fast weight loading method based on weight and model separation to efficiently accelerate inference tasks for PTMs in resource-constrained environments. Furthermore, an online scheduling algorithm is designed to reduce task loading time and inference time. The results of the experiments indicate that FastPTM can improve the processing speed by 8.2 times, reduce the number of switches by 4.7 times, and decrease the number of timeout requests by 15.3 times.

## environment
torch 1.9.0+cu111

## code structure

```
code
├───bert4pytorch    FastPTM framework
│   ├───configs
│   ├───models
│   ├───optimizers
│   ├───tests
│   ├───tokenizers
│   └───trainers
├───models
└───test
    ├───g20
    └───g40
```

## deployment

server：
```
python fastptm_server.py --workers=2
```

client：
```
import fastptm_frame
client = FastPTM_Client()
```

## test

### Impact of number of parallel instances (Figure. 3)
```
python workers.py --task=fcfs_test --machine=1 --num=20
python workers.py --task=fcfs_test --machine=2 --num=20
pythonworkers.py --task=fcfs_test --machine=3 –-num=20
python workers.py --task=fcfs_test --machine=4 --num=20
python workers.py --task=fcfs_test --machine=5 --num=20
python workers.py --task=fcfs_test --machine=6 --num=20
python workers.py --task=fcfs_test --machine=7 --num=20
python workers.py --task=fcfs_test --machine=8 --num=20
Note: A total of 20 groups of multi-tenant requests containing 12 types of tasks in each group.
```

### Impact of fast weight loading (Figure. 4)
  
Turn on MPS:
``` 
export CUDA_VISIBLE_DEVICES=0 nvidia-smi -i 0 -c EXCLUSIVE_PROCESS nvidia-cuda-mps-control -d
``` 
Turn off MPS:
``` 
echo quit | nvidia-cuda-mps-control nvidia-smi -i 0 -c DEFAULT
```
(1) In NVIDIA GTX 1080Ti:
```
cd 1080
./test_switch.sh 1 > nlogs
./test_switch_machine_1.log
./test_switch.sh 2 > nlogs/test_switch_machine_2.log
./test_switch.sh 3 > nlogs/test_switch_machine_3.log
python log2xls.py –task=switch –logfile=./switch_1080.log
Note: log2xls.py is used to extract log results and gen- erate Excel sheet files.
```
(2) In NVIDIA RTX 3090:
```
cd 3090
./test_switch.sh 1 > nlogs/no_mps/test_switch_machine_1.log
./test_switch.sh 2 > nlogs/no_mps/test_switch_machine_2.log
./test_switch.sh 3 > nlogs/no_mps/test_switch_machine_3.log
./test_switch.sh 4 > nlogs/no_mps/test_switch_machine_4.log
./test_switch.sh 5 > nlogs/no_mps/test_switch_machine_5.log
./test_switch.sh 6 > nlogs/no_mps/test_switch_machine_6.log
python log2xls.py –task=switch –logfile=./switch_3090.log
```

### Impact of online multi-list scheduling (Figure. 5 and 6)

(1) In NVIDIA GTX 1080Ti:
```
cd g20
./test_task_1080.sh 202301
python log2xls.py –-task=mu –-logfile=task_mu005_202301.log
python log2xls.py –-task=mu –-logfile=task_mu008_202301.log
python log2xls.py –-task=mu –-logfile=task_mu010_202301.log
```

(2) In NVIDIA RTX 3090:

```
cd code1/g20
./test_task_3090.sh 202302
python log2xls.py –-task=mu –-logfile=task_mu005_202302.log
python log2xls.py –-task=mu –-logfile=task_mu008_202302.log
python log2xls.py –-task=mu –-logfile=task_mu010_202302.log
```
### Overall Performance of FastPTM (Figure.7)

(1) In NVIDIA GTX 1080Ti:

Full model loading:
```
python workers.py –task=fcfs_test –machine=3 –num=20
```

FastPTM:
```
python workers.py –task=omls –machine=3 –taskfile=../task_data/g2/167281892465_dat_T240_L12_M0.10.json
```

(2) In NVIDIA RTX 3090:

Full model loading:
```
python workers.py –task=fcfs_test –machine=6 –num=20
```

FastPTM:
```
python workers.py –task=omls –machine=6 –taskfile= ../task_data/g2/167281892465_dat_T240_L12_M0.10.json
```
