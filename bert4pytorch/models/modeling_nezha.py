#! -*- coding: utf-8 -*-
#
#


import torch
import torch.nn as nn
import copy
from bert4pytorch.layers import LayerNorm, PositionWiseFeedForward, activations, \
    NezhaAttention, NezhaEmbeddings
from bert4pytorch.models.modeling_bert import BertModel


class NezhaLayer(nn.Module):
        def __init__(self, config, ):
        super(NezhaLayer, self).__init__()
        self.multiHeadAttention = NezhaAttention(config)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm1 = LayerNorm(config.hidden_size, eps=1e-12)
        self.feedForward = PositionWiseFeedForward(config)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm2 = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        self_attn_output, layer_attn = self.multiHeadAttention(hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        return hidden_states


class NezhaModel(BertModel):
    
    def __init__(self,
            config,
    ):
        super(NezhaModel, self).__init__(config, )

        self.use_relative_position = config.use_relative_position

        self.embeddings = NezhaEmbeddings(
            config,
        )
        layer = NezhaLayer(
            config,
        )
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        if self.with_pool:
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh()
            if self.with_nsp:
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = activations[self.hidden_act]
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.apply(self.init_model_weights)


    def forward(self, token_ids=None,
                segment_ids=None,
                attention_mask=None,
                output_all_encoded_layers=False):
        
        if attention_mask is None:
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        hidden_states = self.embeddings(token_ids, segment_ids)
        encoded_layers = [hidden_states]
        for layer_module in self.encoderLayer:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, )

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
            )

            if output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        sequence_output = encoded_layers[-1]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state)
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
        else:
            mlm_scores = None
        if mlm_scores is None and nsp_scores is None:
            return encoded_layers, pooled_output
        elif mlm_scores is not None and nsp_scores is not None:
            return mlm_scores, nsp_scores
        elif mlm_scores is not None:
            return mlm_scores
        else:
            return nsp_scores

    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'embeddings.token_type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': 'bert.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': 'bert.embeddings.LayerNorm.bias',
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight',
            'mlmDecoder.bias': 'cls.predictions.decoder.bias'

        }
        for i in range(self.num_hidden_layers):
            prefix = 'bert.encoder.layer.%d.' % i
            mapping.update({'encoderLayer.%d.multiHeadAttention.self.query.weight' % i: prefix + 'attention.self.query.weight',
                            'encoderLayer.%d.multiHeadAttention.self.query.bias' % i: prefix + 'attention.self.query.bias',
                            'encoderLayer.%d.multiHeadAttention.self.key.weight' % i: prefix + 'attention.self.key.weight',
                            'encoderLayer.%d.multiHeadAttention.self.key.bias' % i: prefix + 'attention.self.key.bias',
                            'encoderLayer.%d.multiHeadAttention.self.value.weight' % i: prefix + 'attention.self.value.weight',
                            'encoderLayer.%d.multiHeadAttention.self.value.bias' % i: prefix + 'attention.self.value.bias',
                            'encoderLayer.%d.multiHeadAttention.self.output.weight' % i: prefix + 'attention.output.dense.weight',
                            'encoderLayer.%d.multiHeadAttention.self.output.bias' % i: prefix + 'attention.output.dense.bias',
                            'encoderLayer.%d.layerNorm1.weight' % i: prefix + 'attention.output.LayerNorm.weight',
                            'encoderLayer.%d.layerNorm1.bias' % i: prefix + 'attention.output.LayerNorm.bias',
                            'encoderLayer.%d.feedForward.intermediateDense.weight' % i: prefix + 'intermediate.dense.weight',
                            'encoderLayer.%d.feedForward.intermediateDense.bias' % i: prefix + 'intermediate.dense.bias',
                            'encoderLayer.%d.feedForward.outputDense.weight' % i: prefix + 'output.dense.weight',
                            'encoderLayer.%d.feedForward.outputDense.bias' % i: prefix + 'output.dense.bias',
                            'encoderLayer.%d.layerNorm2.weight' % i: prefix + 'output.LayerNorm.weight',
                            'encoderLayer.%d.layerNorm2.bias' % i: prefix + 'output.LayerNorm.bias'
                            })

        return mapping