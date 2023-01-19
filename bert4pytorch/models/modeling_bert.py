#! -*- coding: utf-8 -*-
#
#

import torch
import torch.nn as nn
import copy
import json
from bert4pytorch.layers import LayerNorm, MultiHeadAttentionLayer, PositionWiseFeedForward, activations


class Transformer(nn.Module):
    
    def __init__(
            self,
            config,
            keep_tokens=None,
            compound_tokens=None,
    ):
        super(Transformer, self).__init__()
        self.vocab_size = config.vocab_size
        if keep_tokens is not None:
            self.vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            self.vocab_size += len(compound_tokens)

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.attention_key_size = self.attention_head_size
        self.intermediate_size = config.intermediate_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_act = config.hidden_act
        self.embedding_size = config.hidden_size
        self.keep_tokens = config.keep_tokens
        self.compound_tokens = config.compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.ignore_invalid_weights = config.ignore_invalid_weights

    def init_model_weights(self, module):
        raise NotImplementedError

    def variable_mapping(self):
                return {}

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        for new_key, old_key in mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict, strict=not self.ignore_invalid_weights)


def lm_mask(segment_ids):
        idxs = torch.arange(0, segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask


def unilm_mask(segment_ids):
    idxs = torch.cumsum(segment_ids, dim=1)
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).to(dtype=torch.float32)
    return mask


class BertEmbeddings(nn.Module):
        def __init__(self, config, ):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids=None, position_ids=None):
        seq_length = token_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
        def __init__(self, config, ):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(config)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm1 = LayerNorm(config.hidden_size, eps=1e-12)
        self.feedForward = PositionWiseFeedForward(config)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm2 = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        self_attn_output, layer_attn = self.multiHeadAttention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        return hidden_states


class BertModel(Transformer):
    
    def __init__(
            self,
            config,
            **kwargs
    ):
        self.initializer_range = config.initializer_range
        self.with_pool = config.with_pool
        self.with_nsp = config.with_nsp
        self.with_mlm = config.with_mlm
        self.custom_position_ids = config.custom_position_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

        super(BertModel, self).__init__(config, **kwargs)

        self.embeddings = BertEmbeddings(
            config,
        )
        layer = BertLayer(
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

    def init_model_weights(self, module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, token_ids=None, token_type_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        
        if attention_mask is None:
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        hidden_states = self.embeddings(token_ids, token_type_ids)
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

            if output_hidden_states:
                encoded_layers.append(hidden_states)
        if not output_hidden_states:
            encoded_layers.append(hidden_states)
        sequence_output = encoded_layers[-1]
        if not output_hidden_states:
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
            'embeddings.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
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
            'mlmDecoder.weight': 'cls.predictions.decoder.weight'

        }
        for i in range(self.num_hidden_layers):
            prefix = 'bert.encoder.layer.%d.' % i
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


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = activations[config.hidden_act]
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


