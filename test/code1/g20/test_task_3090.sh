#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

# 3090  for 1-6 machines
export CUDA_VISIBLE_DEVICES=0

#for i in {1..6} 
#do
#./test_task_mu005_run.sh $i >> 3090/test_task_mu005_202301161710.log
#done
#
#for i in {1..6} 
#do
#./test_task_mu008_run.sh $i >> 3090/test_task_mu008_202301161710.log
#done
#
#for i in {1..6} 
#do
#./test_task_mu010_run.sh $i >> 3090/test_task_mu010_202301161710.log
#done

for j in '005' '008' '010';
do
	for i in {1..6} 
	do
	export runfile=./test_task_mu$j$'_run.sh'
	export logfile=test_task_mu$j$'_202301161710.log'
	$runfile $i >> 3090/$logfile
	done
done



