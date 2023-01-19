#! -*- coding: utf-8 -*-
#
#

import torch
import torch.nn as nn
import copy
import json
from bert4pytorch.layers import LayerNorm, MultiHeadAttentionLayer, PositionWiseFeedForward, activations
from bert4pytorch.models.modeling_bert import BertEmbeddings, BertModel


class RobertaEmbeddings(BertEmbeddings):
        def __init__(self, config, ):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids=None, position_ids=None):
        seq_length = token_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return super(RobertaEmbeddings, self).forward(token_ids,
                                                      segment_ids=segment_ids,
                                                      position_ids=position_ids)


class RobertaModel(BertModel):
    
    def __init__(
            self,
            config,
            **kwargs
    ):
        super(RobertaModel, self).__init__(config, **kwargs)

        self.embeddings = RobertaEmbeddings(
            config,
        )

        self.apply(self.init_model_weights)

    def forward(self, token_ids=None, segment_ids=None, attention_mask=None, output_all_encoded_layers=False, **kwargs):
        if token_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your tokenize.encode()"
                           "or tokenizer.convert_tokens_to_ids().")
        return super(RobertaModel, self).forward(token_ids=token_ids,
                                                 segment_ids=segment_ids,
                                                 attention_mask=attention_mask,
                                                 output_all_encoded_layers=output_all_encoded_layers)

    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'roberta.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': 'roberta.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'roberta.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': 'roberta.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': 'roberta.embeddings.LayerNorm.bias',
            'pooler.weight': 'roberta.pooler.dense.weight',
            'pooler.bias': 'roberta.pooler.dense.bias',
            'mlmDense.weight': 'lm_head.dense.weight',
            'mlmDense.bias': 'lm_head.dense.bias',
            'mlmLayerNorm.weight': 'lm_head.layer_norm.weight',
            'mlmLayerNorm.bias': 'lm_head.layer_norm.bias',
            'mlmBias': 'lm_head.bias',
            'mlmDecoder.weight': 'lm_head.decoder.weight'

        }
        for i in range(self.num_hidden_layers):
            prefix = 'roberta.encoder.layer.%d.' % i
            mapping.update({'encoderLayer.%d.multiHeadAttention.q.weight' % i: prefix + 'attention.self.query.weight',
                            'encoderLayer.%d.multiHeadAttention.q.bias' % i: prefix + 'attention.self.query.bias',
                            'encoderLayer.%d.multiHeadAttention.k.weight' % i: prefix + 'attention.self.key.weight',
                            'encoderLayer.%d.multiHeadAttention.k.bias' % i: prefix + 'attention.self.key.bias',
                            'encoderLayer.%d.multiHeadAttention.v.weight' % i: prefix + 'attention.self.value.weight',
                            'encoderLayer.%d.multiHeadAttention.v.bias' % i: prefix + 'attention.self.value.bias',
                            'encoderLayer.%d.multiHeadAttention.o.weight' % i: prefix + 'attention.output.dense.weight',
                            'encoderLayer.%d.multiHeadAttention.o.bias' % i: prefix + 'attention.output.dense.bias',
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

