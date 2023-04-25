#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动测试
echo 'start batch test...'

# 模型/权重加载对比试验
export CUDA_VISIBLE_DEVICES=0

# usage:
# test_switch_1-6.sh 202304171535

for i in {1..6} 
	do
	export runfile=./test_switch.sh
	export logfile=test_switch_machine_1$i$'_'$1'.log'
	$runfile $i >> nlogs/$logfile
done



