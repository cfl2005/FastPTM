#!/bin/bash
#chkconfig: 2345 80 90
#description: 
echo 'start batch test...'

cd ../

python workers.py --task=omls --machine=$1 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

python workers.py --task=fcfs --machine=$1 --taskfile=../task_data/g4/167296538285_dat_T240_L6_M0.08.json
sleep 5

