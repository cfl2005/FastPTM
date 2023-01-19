#! -*- coding: utf-8 -*-
#
#

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import os
import sys


class BertConfig(object):
    
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 max_relative_position=64,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 attention_scale=True,
                 with_pool=True,
                 with_nsp=False,
                 with_mlm=False,
                 hierarchical_position=False,
                 custom_position_ids=None,
                 keep_tokens=None,
                 compound_tokens=None,
                 is_dropout=False,
                 ignore_invalid_weights=True,
                 return_attention_scores=True,
                 ):
                if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.max_relative_position = max_relative_position
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.initializer_range = initializer_range

            self.attention_scale = attention_scale

            self.with_pool = with_pool
            self.with_nsp = with_nsp
            self.with_mlm = with_mlm
            self.hierarchical_position = hierarchical_position
            self.custom_position_ids = custom_position_ids
            self.keep_tokens = keep_tokens
            self.compound_tokens = compound_tokens

            self.is_dropout = is_dropout
            self.ignore_invalid_weights = ignore_invalid_weights
            self.return_attention_scores = return_attention_scores

        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
                config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
                with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        with open(os.path.join(model_dir, "config.json"), "r", encoding='utf-8') as reader:
            text = reader.read()
        json_object = json.loads(text)

        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value

        for k_, v_ in kwargs.items():
            config.__dict__[k_] = v_

        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
                output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
                return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
                with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())