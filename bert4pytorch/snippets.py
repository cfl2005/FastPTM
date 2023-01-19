#! -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import torch.nn.functional as F
import unicodedata
from typing import List, Dict
from torch.nn import CrossEntropyLoss
from bert4pytorch.losses import FocalLoss, LabelSmoothingCrossEntropy
from torch.utils.data import Dataset


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def truncate_sequences(maxlen, indices, *sequences):
        sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


class DataGenerator(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        raise NotImplementedError()


class DataReader(object):
    def __init__(self, args, traindata, devdata=None, testdata=None):
        self.args = args
        self.traindata = traindata
        self.devdata = devdata
        self.testdata = testdata

    def get_label_distribution(self, total_labels):
        labels_dic = {}
        for label in total_labels:
            labels_dic[label] = labels_dic.get(label, 0) + 1

        total_num = sum(list(labels_dic.values()))
        label_distribution = dict((x,  ((y / total_num)*1000000)//10000 ) for x, y in labels_dic.items())
        sorted_label_distribution = dict(sorted(label_distribution.items(), key=lambda x: -float(x[1])))
        final_label_distribution = {k:str(v)+'%' for k,v in sorted_label_distribution.items()}
        return final_label_distribution

    def load_data(self, index):
        raise NotImplementedError()

    def get_examples(self, add_label=None):
        train = self.args.data_dir + self.traindata
        train_example, train_total_labels, train_label_distribution = self.load_data(train)

        if self.devdata is None and self.testdata is not None:
            test = self.args.data_dir + self.testdata
            test_example, test_total_labels, test_label_distribution = self.load_data(test)

            label_list = list(set(train_total_labels + test_total_labels))
            if add_label is not None:
                label_list += [add_label]
            label_map = {k: i for i, k in enumerate(label_list)}
            num_labels = len(label_list)

            return train_example, train_label_distribution, test_example, test_label_distribution, label_map, num_labels

        if self.testdata is None and self.devdata is not None:
            dev = self.args.data_dir + self.devdata
            dev_example, dev_total_labels, dev_label_distribution = self.load_data(dev)

            label_list = list(set(train_total_labels + dev_total_labels))
            if add_label is not None:
                label_list += [add_label]
            label_map = {k: i for i, k in enumerate(label_list)}
            num_labels = len(label_list)

            return train_example, train_label_distribution, dev_example, dev_label_distribution, label_map, num_labels

        if self.testdata is None and self.devdata is None:
            label_list = list(set(train_total_labels))
            if add_label is not None:
                label_list += [add_label]
            label_map = {k: i for i, k in enumerate(label_list)}
            num_labels = len(label_list)

            return train_example, train_label_distribution, label_map, num_labels

        if self.testdata is not None and self.devdata is not None:
            dev = self.args.data_dir + self.devdata
            dev_example, dev_total_labels, dev_label_distribution = self.load_data(dev)
            test = self.args.data_dir + self.testdata
            test_example, test_total_labels, test_label_distribution = self.load_data(test)

            label_list = list(set(train_total_labels + dev_total_labels + test_total_labels))
            if add_label is not None:
                label_list += [add_label]
            label_map = {k: i for i, k in enumerate(label_list)}
            num_labels = len(label_list)

            return train_example, train_label_distribution, dev_example, dev_label_distribution, test_example,\
                   test_label_distribution, label_map, num_labels



def sequence_padding(inputs, length, value=0):

    outputs = inputs + (length - len(inputs)) * [value]
    return torch.LongTensor(np.array(outputs))


def load_bert_vocab(vocab_path) -> Dict[str, int]:
        with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip("\n")] = index
    return word2idx


def insert_arguments(**arguments):
        def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)
    load_tf_weights_in_bert(model, tf_checkpoint_path)
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


def batch_sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def batch_padding_label_table(labels, length=64):
    
    labels_padded = []
    for lab_table in labels:
        dim_0 = lab_table.shape[0]
        length_1 = lab_table.shape[1]
        lab_table_new = np.zeros((dim_0, length, length))
        for i in range(dim_0):
            for j in range(min(length_1, length)):
                for k in range(min(length_1, length)):
                    lab_table_new[i][j][k] = lab_table[i][j][k]

        labels_padded.append(lab_table_new)

    return labels_padded

def l2_normalize(vecs):
        norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
        import scipy.stats
    return scipy.stats.spearmanr(x, y).correlation


class token_rematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
                        if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
                        return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
                        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
                if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping


def kl_distance(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), -1)
    return torch.mean(_kl)