#!/usr/bin/env python3
#coding:utf-8
dataset_dict = {
                'A':
                    {'dataset': 'lcqmc', 
                    'mtype':'cls',
                    'model_path':'../models/model_lcqmc/model_20220408_044019',
                    'data_path':'../data/LCQMC',
                    },
                'B':
                    {'dataset': 'tnews', 
                    'mtype':'cls',
                    'model_path':'../models/model_tnews/model_20220408_051804',
                    'data_path':'../data/TNews_public',
                    },
                'C':
                    {'dataset': 'os10', 
                    'mtype':'cls',
                    'model_path':'../models/model_cls_os10/model_20220304_034610',
                    'data_path':'../data/online_shopping_10_cats_clsid',
                    },
                'D':
                    {'dataset': 'pdaily', 
                    'mtype':'ner',
                    'model_path':'../models/model_people/model_20220429_070334',
                    'data_path':'../data/pdaily',
                    },
                'E':
                    {'dataset': 'choice', 
                    'mtype':'ner',
                    'model_path':'../models/model_people/model_20220429_070334',
                    'data_path':'../data/choice',
                    },
                'F':
                    {'dataset': 'cmrc', 
                    'mtype':'ner',
                    'model_path':'../models/model_people/model_20220429_070334',
                    'data_path':'../data/choice',
                    },
                'G':
                    {'dataset': 'lcqmc', 
                    'mtype':'cls',
                    'model_path':'../models/model_lcqmc/model_20220408_044019',
                    'data_path':'../data/LCQMC',
                    },
                'H':
                    {'dataset': 'tnews', 
                    'mtype':'cls',
                    'model_path':'../models/model_tnews/model_20220408_051804',
                    'data_path':'../data/TNews_public',
                    },
                'I':
                    {'dataset': 'choice', 
                    'mtype':'ner',
                    'model_path':'../models/model_people/model_20220429_070334',
                    'data_path':'../data/choice',
                    },
                'J':
                    {'dataset': 'cmrc', 
                    'mtype':'ner',
                    'model_path':'../models/model_people/model_20220429_070334',
                    'data_path':'../data/choice',
                    },
                'K':
                    {'dataset': 'lcqmc', 
                    'mtype':'cls',
                    'model_path':'../models/model_lcqmc/model_20220408_044019',
                    'data_path':'../data/LCQMC',
                    },
                'L':
                    {'dataset': 'tnews', 
                    'mtype':'cls',
                    'model_path':'../models/model_tnews/model_20220408_051804',
                    'data_path':'../data/TNews_public',
                    },
               }
gbl_model_counts = len(dataset_dict)
gbl_model_chars = ''.join(dataset_dict.keys())
table_model_loaded = [4] * gbl_model_counts     
table_model_loaded = [400] * gbl_model_counts
table_model_loaded = [150] * gbl_model_counts
table_model_predict_speed = [100] * gbl_model_counts
table_model_predict_speed = [80] * gbl_model_counts
table_model_predict_speed = [180] * gbl_model_counts
table_model_predict_speed = [8] * gbl_model_counts
table_model_deadline = [3] * gbl_model_counts
switch_threshold = 50
switch_threshold = 1000
switch_threshold = 800
class conf_lcqmc:
    task_type = 'A'
    SEED = 42
    learning_rate = 4e-4
    num_train_epochs = 1
    max_len = 128
    batch_size = 512
    num_labels = 2

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "lcqmc"
    data_dir = "../../code/data/lcqmc/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/lcqmc/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_tnews:
    task_type = 'B'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 15

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "tnews"
    data_dir = "../../code/data/tnews/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/tnews/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"


class conf_os10:
    task_type = 'C'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 3

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'

    dataset_name = "os10"
    data_dir = "../../code/data/os10/"

    model_outpath = "../outputs/os10/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"


class conf_pdaily:
    task_type = 'D'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 3
    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "pdaily"
    data_dir = "../../code/data/pdaily/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/pdaily/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_choice:
    task_type = 'E'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 3

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'

    dataset_name = "choice"
    data_dir = "../../code/data/choice/"

    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/choice/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_cmrc:
    task_type = 'F'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 3

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'

    dataset_name = "cmrc"
    data_dir = "../../code/data/cmrc/"

    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/cmrc/"
    save_path = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_lcqmc1:
    task_type = 'G'
    SEED = 42
    learning_rate = 4e-4
    num_train_epochs = 1
    max_len = 128
    batch_size = 512
    num_labels = 2

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "lcqmc"
    data_dir = "../../code/data/lcqmc/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/lcqmc/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_tnews1:
    task_type = 'H'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 15

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "tnews"
    data_dir = "../../code/data/tnews/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/tnews/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"


class conf_lcqmc2:
    task_type = 'I'
    SEED = 42
    learning_rate = 4e-4
    num_train_epochs = 1
    max_len = 128
    batch_size = 512
    num_labels = 2

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "lcqmc"
    data_dir = "../../code/data/lcqmc/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/lcqmc/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_tnews2:
    task_type = 'J'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 15

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "tnews"
    data_dir = "../../code/data/tnews/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/tnews/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"


class conf_lcqmc3:
    task_type = 'K'
    SEED = 42
    learning_rate = 4e-4
    num_train_epochs = 1
    max_len = 128
    batch_size = 512
    num_labels = 2

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "lcqmc"
    data_dir = "../../code/data/lcqmc/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/lcqmc/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

class conf_tnews3:
    task_type = 'L'
    SEED = 42
    learning_rate = 4e-5
    num_train_epochs = 5
    max_len = 128
    batch_size = 512
    num_labels = 15

    model_name = "bert"
    model_path = '/mnt/sda1/models/pytorch/chinese_wwm_pytorch'
    dataset_name = "tnews"
    data_dir = "../../code/data/tnews/"
    model_outpath = "/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/tnews/"
    save_path = f"/mnt/sda1/transdat/ext_job/SGT_Project/pytorch/outputs/{model_name}_{dataset_name}_{SEED}"

all_models = [conf_lcqmc(),
              conf_tnews(),  
              conf_lcqmc(),  
              conf_lcqmc(),  
              conf_lcqmc(),  
              conf_lcqmc(),  
              conf_lcqmc1(), 
              conf_tnews1(), 
              conf_lcqmc2(), 
              conf_tnews2(), 
              conf_lcqmc3(), 
              conf_tnews3(), 
              ]
class_model_dict = dict(zip(gbl_model_chars, all_models))



if __name__ == '__main__':
    pass

