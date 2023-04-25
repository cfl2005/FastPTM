#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动测试
echo 'start batch test...'

# 模型/权重加载对比试验

cd ../

# 模型切换 40组
#python workers.py --task=worker_switch --machine=$1 --num=40 --task_list=ABCDEF --isfull=1
python workers.py --task=worker_switch --machine=$1 --num=$2 --isfull=$3
sleep 5





