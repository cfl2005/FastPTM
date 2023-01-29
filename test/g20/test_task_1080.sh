#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

# 1080 for 1-3机器
export CUDA_VISIBLE_DEVICES=1

#for i in {1..3} 
#do
#./test_task_mu005_run.sh $i >> 1080/test_task_mu005_202301161706.log
#done
#
#for i in {1..3} 
#do
#./test_task_mu008_run.sh $i >> 1080/test_task_mu008_202301161706.log
#done
#
#for i in {1..3} 
#do
#./test_task_mu010_run.sh $i >> 1080/test_task_mu010_202301161706.log
#done

for j in '005' '008' '010'
do 
	for i in {1..3} 
	do
	export runfile=./test_task_mu$j$'_run.sh'
	export logfile=test_task_mu$j$'_202301161706.log'
	$runfile $i >> 1080/$logfile
	done
done

