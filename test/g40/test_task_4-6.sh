#!/bin/bash
#chkconfig: 2345 80 90
#description: Test
echo 'start batch test...'

# 3090 for 4-6 machines

for i in {4..6} 
do
./test_task_mu005_run.sh $i >> 3090/test_task_mu005_202301140040.log
./test_task_mu008_run.sh $i >> 3090/test_task_mu008_202301140040.log
./test_task_mu010_run.sh $i >> 3090/test_task_mu010_202301140040.log
done
