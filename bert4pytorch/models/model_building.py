#! -*- coding: utf-8 -*-


import os
import torch
import torch.nn as nn
from bert4pytorch.models.modeling_bert import BertModel
from bert4pytorch.models.modeling_nezha import NezhaModel
from bert4pytorch.models.modeling_roberta import RobertaModel

def build_transformer_model(
        config=None,
        model_path=None,
        model_name=None,
):
        if model_name == None:
        try:
            model_name = config.model_type
        except AttributeError:
            raise AttributeError('Please pass the model_name parameter!')

    models = {
        'bert': BertModel,
        'roberta': RobertaModel,
        'nezha': NezhaModel,
    }

    my_model = models[model_name]
    transformer_model = my_model(config)
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    transformer_model.load_weights_from_pytorch_checkpoint(checkpoint_path)

    return transformer_model
def save_model(model, output_model_file):
    state_dict = model.state_dict()
    torch.save(state_dict, output_model_file)
class BertBase(nn.Module):
    def __init__(self, config, model_path, model_name='bert', **kwargs):
        super(BertBase, self).__init__()
        config.with_pool = False
        self.bert = build_transformer_model(
            config=config,
            model_path=model_path,
            model_name=model_name
        )

    def forward(self, token_ids=None, segment_ids=None, attention_mask=None, **inputs):
        output, _ = self.bert(token_ids=token_ids, 
                                        segment_ids=segment_ids, 
                                        attention_mask=attention_mask)
        return output
class model_task_classify(nn.Module):
    def __init__(self, config):
        super(model_task_classify, self).__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.fc = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, sequence_output):
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        logits = self.fc(pooled_output)
        return logits


class Bert4TorchModel(nn.Module):
    def __init__(self, config, model_path, model_name, **kwargs):
        super(Bert4TorchModel, self).__init__()
        self.bert = build_transformer_model(
            config=config,
            model_path=model_path,
            model_name=model_name
        )
        self.config = config
        self.dropout = nn.Dropout(p=0.3)

        try:
            self.fc = nn.Linear(config.hidden_size, config.num_labels)
        except AttributeError:
            print('BertConfig object has no attribute num_labels')

    def forward(self, **kwargs):
        raise NotImplementedError()


class Model_tnews(nn.Module):
    def __init__(self, config, model_path, model_name, **kwargs):
        super(Model_tnews, self).__init__()

        config.with_pool = False

        self.bert = build_transformer_model(
            config=config,
            model_path=model_path,
            model_name=model_name
        )
        self.config = config
        self.task = model_task_classify(config)

    def forward(self, token_ids=None, segment_ids=None, attention_mask=None, **inputs):
        sequence_output = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        logits = self.task(sequence_output[0])
        return logits
class Model_lcqmc(nn.Module):
    def __init__(self, config, model_path, model_name, **kwargs):
        super(Model_lcqmc, self).__init__()
        config.with_pool = False
        self.bert = build_transformer_model(
            config=config,
            model_path=model_path,
            model_name=model_name
        )
        self.config = config
                self.task = model_task_classify(config)

    def forward(self, token_ids=None, segment_ids=None, attention_mask=None, **inputs):
        sequence_output = self.bert(token_ids=token_ids, segment_ids=segment_ids, attention_mask=attention_mask)
        logits = self.task(sequence_output[0])
        return logits
class model_task_ner(nn.Module):
    def __init__(self, config):
        super(model_task_classify, self).__init__()

        self.fc = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, sequence_output):
        logits = self.fc(sequence_output)
        return logits
class Model_NER(nn.Module):
    def __init__(self, config, model_path, model_name, **kwargs):
        super(Model_lcqmc, self).__init__()
        config.with_pool = False
        self.bert = build_transformer_model(
            config=config,
            model_path=model_path,
            model_name=model_name
        )
        self.config = config
        self.task = model_task_classify(config)

    def forward(self, token_ids=None, segment_ids=None, attention_mask=None, **inputs):
        sequence_output = self.bert(token_ids=token_ids, segment_ids=segment_ids, attention_mask=attention_mask)
        logits = self.task(sequence_output[0])
        return logits



