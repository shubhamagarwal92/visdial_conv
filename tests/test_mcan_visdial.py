# Run as PYTHONPATH=. python tests/test_mcan_visdial.py

from visdialch.encoders.mcan_img_only import MCANImgOnlyEncoder
from visdialch.data.vocabulary import Vocabulary

import torch

def test_encoder():

    # a = torch.randn(2,5)
    # b = torch.randn(2,5)
    # c = a+b
    # print(a)
    # print(b)
    # print(c)

    # scores = scores.masked_fill(mask, -1e9)

    img_feat = torch.randn(4,36,2048)
    seq_size = 20
    ques = torch.randperm(seq_size).view(1, seq_size)  # (batch,seq_len)
    ques = ques.unsqueeze(1).repeat(4,10,1)
    # ques = ques.repeat(4,1)
    ques_len = torch.LongTensor([6,5,4,3]).unsqueeze(1).repeat(1,10)
    # print(ques_len.size())
    #
    # print(ques.size())  # (4,10,20)
    # print(img_feat.size())

    config = {
        "use_hist": False,
        "use_bert": False,
        "img_feature_size": 2048,
        "word_embedding_size": 300,
        "bert_embedding_size": 768,
        "lstm_hidden_size": 512,
        "lstm_num_layers": 2,
        "dropout": 0.5,
        "word_counts_json": '../data/visdial_1.0_word_counts_train.json',
        "concat_history": False,
        "vocab_min_count": 5
    }

    vocabulary = Vocabulary(
                config["word_counts_json"], min_count=config["vocab_min_count"])

    net = MCANImgOnlyEncoder(config, vocabulary)
    opts = {
        'img_feat': img_feat,
        'ques': ques,
        'ques_len': ques_len
    }

    fused_embedding = net(opts)
    print(fused_embedding.size())

    # With history, not concatenated

    # print("With history concat false")
    # config["use_hist"] = True
    # net = MCANImgOnlyEncoder(config, vocabulary)
    # hist = ques
    # hist_len = ques_len
    # opts = {
    #     'img_feat': img_feat,
    #     'ques': ques,
    #     'ques_len': ques_len,
    #     'hist': hist,
    #     'hist_len': hist_len
    #
    # }

    # fused_embedding = net(opts)
    # print(fused_embedding.size())


test_encoder()
