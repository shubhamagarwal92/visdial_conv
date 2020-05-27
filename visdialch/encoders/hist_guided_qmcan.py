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

class HistGuidedQMCANEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary

        self.mcan_config = Cfgs()

        # SA: Overwriting fixed configs in MCAN by config parameters
        self.img_mcan_config = self.mcan_config
        self.img_mcan_config.MULTI_HEAD = config.get("qi_multi_head", 8)
        self.img_mcan_config.LAYER = config.get("qi_attn_layer", 6)

        # ans embedding size
        self.image_MCAN_Net = MCAN_Net(self.img_mcan_config, answer_size=config["lstm_hidden_size"])

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

        # self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly

        if self.use_hist:

            self.qh_mcan_config = self.mcan_config
            self.qh_mcan_config.MULTI_HEAD = config.get("qi_multi_head", 8)
            self.qh_mcan_config.LAYER = config.get("qi_attn_layer", 6)

            self.hist_MCAN_Net = MCAN_Net(self.qh_mcan_config, answer_size=config["lstm_hidden_size"],
                                          only_return_y=True)

            self.hist_rnn = nn.LSTM(
                self.word_embed_size_for_rnn,
                config["lstm_hidden_size"],
                config["lstm_num_layers"],
                batch_first=True,
                dropout=config["dropout"]
            )
            self.hist_rnn = DynamicRNN(self.hist_rnn)


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

        ques = ques.squeeze(1) ## Hacky way to get around masking
        ques_embed = self.word_embed(ques)

        # shape: (batch_size * num_rounds, max_sequence_length,
        #         lstm_hidden_size)
        q_embed_per_word, (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])


        assert self.use_hist == True, "history should be there for guided attention"
        # if concat history:
        # shape: (batch_size, 10, max_sequence_length * 2 * 10)
        # concatenated qa * 10 rounds
        # else:
        # (batch_size, 10, max_sequence_length * 2)  (#QA combined)
        hist = batch["hist"]

        # embed history
        # SA: removing hard-coding to allow mn
        # hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist = hist.view(batch_size * num_rounds, -1)

        # SA: removing this now
        lang_hist_feat_mask = make_mask(hist.unsqueeze(2))

        hist_embed = self.word_embed(hist)

        # shape: (batch_size * num_rounds, lstm_hidden_size)
        h_embed_per_word, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])

        # SA: be careful - now we are enhancing question based on history embeddings

        # (bs*num_rounds, lstm_hidden_size)
        self_attnd_hist, enhanced_q = self.hist_MCAN_Net(q_embed_per_word, h_embed_per_word,
                                           lang_hist_feat_mask, return_sep_modes=True)


        # Make mask using the original question only
        ques_feat_mask = make_mask(ques.unsqueeze(2))

        # (batch_size * num_rounds, lstm_hidden_size)
        img_ques_mcan_embedding = self.image_MCAN_Net(image_features, enhanced_q, ques_feat_mask)

        # shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = img_ques_mcan_embedding.view(batch_size, num_rounds, -1)

        return fused_embedding
