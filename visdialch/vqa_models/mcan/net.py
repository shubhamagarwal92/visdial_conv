# Taken from https://github.com/MILVLG/openvqa/blob/master/openvqa/models/mcan/net.py

from visdialch.vqa_models.mcan.make_mask import make_mask
from visdialch.vqa_models.mcan.fc import FC, MLP
from visdialch.vqa_models.mcan.layer_norm import LayerNorm
from visdialch.vqa_models.mcan.mca import MCA_ED
from visdialch.vqa_models.mcan.model_cfgs import Cfgs
import torch.nn as nn
import torch.nn.functional as F
import torch
import typing

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        # FLAT_GLIMPSES == 1
        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        # print(x.size())  # (bs*round, length, emb size=512)
        att = self.mlp(x)
        # print(att.size())  # (bs*round, length, 1)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        # print(att.size())  # (bs*round, length, 1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        #  MLP attention wise sum for each example --> note dim is 1 here

        # z = torch.sum(att[:, :, i: i + 1] * x, dim=1)
        # print(z.size())  # (bs*round, emb_size)

        x_atted = torch.cat(att_list, dim=1)
        # print(x_atted.size())  # (bs*round, emb size=512)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class MCAN_Net(nn.Module):
    def __init__(self, __C, answer_size,
                 only_return_y:bool = False):
        super().__init__()
        self.__C = __C

        # SA: try this now
        # self.__C = Cfgs()

        # self.embedding = nn.Embedding(
        #     num_embeddings=token_size,
        #     embedding_dim=__C.WORD_EMBED_SIZE
        # )

        # Loading the GloVe embedding weights
        # if __C.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # self.lstm = nn.LSTM(
        #     input_size=__C.WORD_EMBED_SIZE,
        #     hidden_size=__C.HIDDEN_SIZE,
        #     num_layers=1,
        #     batch_first=True
        # )

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.only_return_y = only_return_y


    # def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
    def forward(self, img_feat, lang_feat,
                lang_feat_mask, return_sep_modes=False):

        """

        Y: img_feat
        X: lang_feat

        :param img_feat: same embedding size as ques  (bs, num_boxes, 512/1024)
        :param ques_ix:  (bs, max_length, 512/1024)
        :return:
        """

        # Pre-process Language Feature

        # lang_feat_mask = make_mask(text)


        # lang_feat = self.embedding(ques_ix)
        # lang_feat, _ = self.lstm(lang_feat)

        # print("Lang feat", lang_feat.size())  # ([bs, seq len, hidden size]
        # print("lang feat mask", lang_feat_mask.size())  # (

        # img_feat_mask = torch.cat((make_mask(frcn_feat), make_mask(grid_feat)), dim=-1)
        # img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)


        img_feat_mask = make_mask(img_feat)

        # print(lang_feat.size())
        # print(lang_feat_mask.size())
        # print(img_feat_mask.size())
        # print(img_feat.size())


        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # print("Language feature", lang_feat.size())  # (bs*round, length, emb size=512)
        # print("Image feature", img_feat.size())  # (bs*round, proposal, emb size=512)
        #
        if return_sep_modes:
            return lang_feat, img_feat


        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        # print(lang_feat.size())  # Flattened vector now (bs*round, 1024)

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # print("Language feature", lang_feat.size())  # (bs*round, 1024)
        # print("Image feature", img_feat.size())  # (bs*round, 1024)

        # SA: we dont want to concat these features
        if self.only_return_y:
            return img_feat

        # Element wise sum
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)  # (bs, 512)

        return proj_feat

