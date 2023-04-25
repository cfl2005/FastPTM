#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动测试
echo 'start batch test...'
export CUDA_VISIBLE_DEVICES=1

# 全系列测试命令脚本
cd ../

# 12模型 1机20组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

# 12模型 1机40组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

# 6模型 1机20组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

# 6模型 1机40组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

# 12模型 2机20组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

# 12模型 2机40组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

# 6模型 2机20组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

# 6模型 2机40组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

# 12模型 3机20组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g4/167296538325_dat_T240_L12_M0.08.json
sleep 5

# 12模型 3机40组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g4/167296538370_dat_T480_L12_M0.08.json
sleep 5

# 6模型 3机20组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g4/167296538219_dat_T120_L6_M0.08.json
sleep 5

# 6模型 3机40组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5
