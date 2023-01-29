#!/bin/bash
#chkconfig: 2345 80 90
#description: 
echo 'start batch test...'


cd ../

python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=omls --machine=1 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=1 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=omls --machine=2 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=2 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892465_dat_T480_L12_M0.10.json
sleep 5

python workers.py --task=omls --machine=3 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5

python workers.py --task=fcfs --machine=3 --taskfile=../task_data/g2/167281892385_dat_T240_L6_M0.10.json
sleep 5
