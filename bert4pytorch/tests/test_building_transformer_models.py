#! -*- coding: utf-8 -*-
#
#

import sys
sys.path.append("./")

from bert4pytorch.configs.configuration_bert import BertConfig
from bert4pytorch.models.model_building import build_transformer_model


if __name__ == "__main__":
    #
    model_path = "./resources/chinese_bert_wwm_ext"
    kwargs = {
        "with_pool": True,
        "max_sequence": 128,
    }
    config = BertConfig.from_pretrained(
        model_path,
        **kwargs
    )

    print("config: ", config)
    build_transformer_model(
        config=config,
        model_path=model_path,
        model_name='bert',
        **kwargs
    )