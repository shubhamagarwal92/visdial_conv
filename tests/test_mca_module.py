# Run as PYTHONPATH=. python tests/test_mca_module.py

from visdialch.vqa_models.mcan.net import MCAN_Net
from visdialch.vqa_models.mcan.model_cfgs import Cfgs
from visdialch.vqa_models.mcan.make_mask import make_mask

import torch


def test_encoder():
    img_feat = torch.randn(4,36,2048)
    seq_size = 20
    ques = torch.randperm(seq_size).view(1, seq_size)  # (batch,seq_len)
    ques = ques.unsqueeze(1).repeat(4,10,1)
    # ques = ques.repeat(4,1)
    ques_len = torch.LongTensor([6,5,4,3]).unsqueeze(1).repeat(1,10)
    # print(ques_len.size())



if __name__=="__main__":
    test_encoder()
