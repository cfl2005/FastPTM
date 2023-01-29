#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

cd ../

# 6 Model  N machines 40 Groups
python workers.py --task=omls --machine=$1 --taskfile=../task_data/g3/167288556111_dat_T240_L6_M0.05.json
sleep 5

python workers.py --task=fcfs --machine=$1 --taskfile=../task_data/g3/167288556111_dat_T240_L6_M0.05.json
sleep 5

