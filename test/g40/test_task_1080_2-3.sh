#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

# 1080 for 2-3 machines

export CUDA_VISIBLE_DEVICES=1

for j in '005' '008' '010'
do 
	for i in {2..3} 
	do
	export runfile=./test_task_mu$j$'_m6_run.sh'
	export logfile=test_task_mu$j$'_202301191140.log'
	$runfile $i >> 1080/0119/$logfile
	done
done
