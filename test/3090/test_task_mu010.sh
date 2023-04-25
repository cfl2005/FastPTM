#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动测试
echo 'start batch test...'

# 全系列测试命令脚本

cd ../

# 12模型 1机20组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

# 12模型 1机40组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

# 6模型 1机20组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

# 6模型 1机40组
python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

# 12模型 2机20组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

# 12模型 2机40组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

# 6模型 2机20组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

# 6模型 2机40组
python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

# 12模型 3机20组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

# 12模型 3机40组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

# 6模型 3机20组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892345_dat_T120_L6_M0.10.json
sleep 5

# 6模型 3机40组
python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5
