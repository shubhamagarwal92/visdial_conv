# SA: todo NOTE be careful about config for mcan
## SA:

# MCAN  -- image features always project

import torch
from torch import nn
from torch.nn import functional as F

from visdialch.utils import DynamicRNN
from visdialch.vqa_models.mcan.net import MCAN_Net
from visdialch.vqa_models.mcan.model_cfgs import Cfgs
from visdialch.vqa_models.mcan.make_mask import make_mask
from visdialch.utils import dotdict

class MCANImgMCANVQAHistAttnEncoder(nn.Module):
    def __init__(self, config, vocabulary, num_rounds=10):
        """
        SA: TODO we have hardcoded num_rounds for now to 10.
        We have defined mask in init for speed computation as it
        is static. Ideally should be in forward. need a better
        way to masking

        :param config:
        :param vocabulary:
        :param num_rounds:
        """
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary

        self.mcan_config = Cfgs()
        # ans embedding size
        self.image_MCAN_Net = MCAN_Net(self.mcan_config, answer_size=config["lstm_hidden_size"])

        self.use_hist = config.get("use_hist", False)

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.word_embed_size_for_rnn = config["word_embedding_size"]

        self.ques_rnn = nn.LSTM(
            self.word_embed_size_for_rnn,
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True)
        # SA: removing dropout for mcan

        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly

        if self.use_hist:
            self.hist_MCAN_Net = MCAN_Net(self.mcan_config, answer_size=config["lstm_hidden_size"])

            self.hist_rnn = nn.LSTM(
                self.word_embed_size_for_rnn,
                config["lstm_hidden_size"],
                config["lstm_num_layers"],
                batch_first=True,
                dropout=config["dropout"]
            )
            self.hist_rnn = DynamicRNN(self.hist_rnn)


            self.vqa_MCAN_Net = MCAN_Net(self.mcan_config, answer_size=config["lstm_hidden_size"])

            self.mask_prev_rounds_ = self.mask_prev_rounds(num_rounds=num_rounds,
                                                           emb_size=config["lstm_hidden_size"])


        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # project image features to lstm_hidden_size for computing attention
        self.image_features_projection = nn.Linear(
            config["img_feature_size"], config["lstm_hidden_size"]
        )

        fusion_size = (
            config["lstm_hidden_size"] * 2
        )
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # num_rounds = 10, even for test (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()

        # project down image features and ready for attention
        # shape: (batch_size, num_proposals, lstm_hidden_size)
        projected_image_features = self.image_features_projection(img)
        # repeat image feature vectors to be provided for every round
        # shape: (batch_size * num_rounds, num_proposals, lstm_hidden_size)
        projected_image_features = (
            projected_image_features.view(
                batch_size, 1, -1, self.config["lstm_hidden_size"]
            )
            .repeat(1, num_rounds, 1, 1)
            .view(batch_size * num_rounds, -1, self.config["lstm_hidden_size"])
        )

        image_features = projected_image_features

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)

        lang_feat_mask = make_mask(ques.unsqueeze(2))

        ques_embed = self.word_embed(ques)

        # shape: (batch_size * num_rounds, max_sequence_length,
        #         lstm_hidden_size)
        q_embed_per_word, (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])

        device = ques_embed.device
        # (batch_size * num_rounds,
        #         lstm_hidden_size)
        img_ques_mcan_embedding = self.image_MCAN_Net(image_features, q_embed_per_word, lang_feat_mask)

        # if concat history:
        # shape: (batch_size, 10, max_sequence_length)
        # concatenated qa * 10 rounds
        # else:
        # (batch_size, 10, max_sequence_length * 2)  (#QA combined)
        hist = batch["hist"]
        # embed history
        hist = hist.view(batch_size * num_rounds, -1)

        hist_feat_mask = make_mask(hist.unsqueeze(2))

        # SA: removing hard-coding to allow mn
        # hist = hist.view(batch_size * num_rounds, max_sequence_length)
        hist_embed = self.word_embed(hist)

        # shape: (batch_size * num_rounds, seq_len, lstm_hidden_size)
        h_embed_per_word, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])

        hist_img_mcan_embedding = self.hist_MCAN_Net(image_features, h_embed_per_word, hist_feat_mask)
        # (bs*round, emb_size) = > (bs, rounds, emb_size)
        hist_img_mcan_embedding = hist_img_mcan_embedding.view(batch_size, num_rounds, -1)

        # Now convert it for all ques and do masking.
        hist_img_mcan_embedding = (
            hist_img_mcan_embedding.unsqueeze(1)
            .repeat(1, num_rounds, 1, 1)
            .view(batch_size * num_rounds, num_rounds, -1)
        )

        # This mask is for one example. Repeat it batch size times
        mask_prev_rounds_ = self.mask_prev_rounds_.repeat(batch_size, 1, 1).to(device).detach()
        hist_img_mcan_embedding = hist_img_mcan_embedding * mask_prev_rounds_

        attended_vqa_hist = self.vqa_MCAN_Net(hist_img_mcan_embedding, q_embed_per_word, lang_feat_mask)


        fused_vector = torch.cat((img_ques_mcan_embedding, attended_vqa_hist), 1)
        fused_vector = self.dropout(fused_vector)

        fused_embedding = self.fusion(fused_vector)

        # Trying w/o non-linearity now
        # fused_embedding = torch.tanh(mcan_fused_rep)

        # shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)

        return fused_embedding


    @staticmethod
    def mask_prev_rounds(num_rounds, emb_size):
        mask_ = torch.ones((num_rounds, num_rounds, emb_size))
        # Create a mask for history
        for i in range(num_rounds):
            mask_[i, i+1:num_rounds, :] = 0
        return mask_
