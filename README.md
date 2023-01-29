# FastPTM : Fast Weight Loading for Inference Acceleration of Pre-Trained Models in GPUs
Large-scale pre-trained models (PTMs) have achieved great success on a wide range of NLP and CV tasks and have become milestones in the field of deep learning (DL). Despite the effectiveness of PTMs, PTMs are typically associated with large memory and high computational requirements, which increase the cost and time of inference. To enable large-scale deployment and real-time response of PTM applications, we propose the FastPTM framework, a generic framework for accelerating the inference tasks of PTMs, deployed in edge GPUs. In the framework, we implement a fast weight loading method based on weight and model separation to efficiently accelerate inference tasks for large-size models in resource-constrained environments. In addition, we design an online scheduling algorithm to reduce the task loading time and inference time. Experiment results show that FastPTM can improve 8.2 times of processing speed and reduces the number of switches by 4.5 times and the number of timeout requests by 13 times.

##environment
torch 1.9.0+cu111

##code structure

```
code
├───bert4pytorch    framework
│   ├───configs
│   ├───models
│   ├───optimizers
│   ├───tests
│   ├───tokenizers
│   └───trainers
├───models	
```
##test
1080Ti：

```
cd code/test/g20/
./test_task_1080.sh

cd code/test/g40/
./test_task_1080_2-3.sh
```


3090Ti：

```
cd code/test/g20/
./test_task_3090.sh
```


After the test is completed, the test data is automatically extracted using the script：

```
cd code/test

python log2xls.py --task=mu --logfile=g20/1080/***.log
python log2xls.py --task=mu --logfile=g40/1080/***.log

python log2xls.py --task=mu --logfile=g20/3090/***.log
python log2xls.py --task=mu --logfile=g40/3090/***.log
```
replace "***" in the above code with the actual generated file name.

The data extracted from the test data will be automatically saved as an EXCEL file with the same name (***.XLS)

##deployment

server：
```
python fastptm_server.py --workers=2
```

client：
```
import fastptm_frame
client = FastPTM_Client()
```
