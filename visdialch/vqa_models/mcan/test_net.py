import torch
import sys
sys.path.append("..")
# from visdialch.vqa_models.mcan.net import Net
# from visdialch.vqa_models.mcan.model_cfgs import Cfgs
from net import MCAN_Net
from model_cfgs import Cfgs



def test_net():
    config = Cfgs()
    print(config)
    token_size = 100
    answer_size = 512

    vocab_size = 100
    seq_size = 10
    ques_ix = torch.randperm(seq_size).view(1, seq_size)  # (batch,seq_len)
    ques_ix = ques_ix.repeat(4,1)
    print(ques_ix)

    net = MCAN_Net(config, pretrained_emb=None, token_size=token_size, answer_size=answer_size)

    img_feat = torch.randn(4,36,512)
    # ques_ix = torch.rand(1,10)
    rep = net(img_feat,ques_ix)
    print(rep.size())


test_net()
# if __name__ == "__main__":
#     test_net()
