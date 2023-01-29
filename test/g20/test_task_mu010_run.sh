#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

cd ../

# 12 Model  N machines 20 Groups
python workers.py --task=omls --machine=$1 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=$1 --taskfile=../task_data/g2/167281892426_dat_T240_L12_M0.10.json
sleep 5
