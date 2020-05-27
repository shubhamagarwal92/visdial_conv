import argparse
from tqdm import tqdm
from typing import Any, Dict
import json
import pickle
import numpy as np
import pandas as pd
import glob
import os
from typing import List
from visdialch.metrics import NDCG
from pathlib import Path
import torch

class NDCGForRanks:
    def __init__(self, config):
        super().__init__()
        self.ndcg = NDCG(is_direct_ranks=True)
        # We are calculating NDCG directly based on ranks
        # self.path_val_data = config.path_val_data
        self.dense_annotations_jsonpath = config.dense_annotations_jsonpath
        self.model_preds_root = config.model_preds_root
        self.models_list = self.get_model_type_list(self.model_preds_root)
        self.annotations_json = json.load(open(self.dense_annotations_jsonpath))
        self.subset_type = config.subset_type

    def extract_ndcg(self, model_type: str):
        """
        :param model_type: for which we want to extract
        :return:
        """
        try:
            ranks_phase1_file = f"ranks_val_12{self.subset_type}.json"
            ranks_finetune_file = f"ranks_val_best_ndcg{self.subset_type}.json"
            ranks_phase1_json, ranks_finetune_json = self.get_both_phase_ranks(model_type,
                                                                               ranks_phase1_file=ranks_phase1_file,
                                                                               ranks_finetune_file=ranks_finetune_file)
        except:
            print(f"For Model {model_type}, we are using 11 as the ckpt based on ndcg")
            ranks_phase1_file = f"ranks_val_11{self.subset_type}.json"
            ranks_finetune_file = f"ranks_val_best_ndcg{self.subset_type}.json"
            ranks_phase1_json, ranks_finetune_json = self.get_both_phase_ranks(model_type,
                                                                               ranks_phase1_file=ranks_phase1_file,
                                                                               ranks_finetune_file=ranks_finetune_file)

        ranks_phase1_json = self.subset_val_ranks_with_dense_annotation(
                            ranks_phase1_json)
        ranks_finetune_json = self.subset_val_ranks_with_dense_annotation(
                              ranks_finetune_json)
        assert len(ranks_phase1_json) == len(ranks_finetune_json) == len(self.annotations_json)
        gt_relevance_list = []
        ranks_list_phase1 = []
        ranks_list_finetune = []
        ndcg_list_phase1 = []
        ndcg_list_finetune = []
        for indx in range(len(ranks_phase1_json)):
            ranks_phase1 = ranks_phase1_json[indx]["ranks"]
            ranks_finetune = ranks_finetune_json[indx]["ranks"]
            gt_relevance = self.annotations_json[indx]['gt_relevance']
            # Assert if we are doing for same round ids
            assert ranks_phase1_json[indx]['round_id'] == self.annotations_json[indx]['round_id']
            assert ranks_finetune_json[indx]['round_id'] == self.annotations_json[indx]['round_id']
            ndcg_sample_phase1 = self.get_ndcg_value_wrapper(ranks_phase1, gt_relevance)
            ndcg_sample_finetune = self.get_ndcg_value_wrapper(ranks_finetune, gt_relevance)
            # Maintain the list for individual samples
            ndcg_list_phase1.append(ndcg_sample_phase1)
            ndcg_list_finetune.append(ndcg_sample_finetune)
            # For whole set - for verification
            ranks_list_phase1.append(ranks_phase1)
            ranks_list_finetune.append(ranks_finetune)
            gt_relevance_list.append(gt_relevance)
        # For whole val set, for verification..Not saving them!
        ndcg_sample_phase1 = self.get_ndcg_value_wrapper(ranks_list_phase1, gt_relevance_list)
        ndcg_sample_finetune = self.get_ndcg_value_wrapper(ranks_list_finetune, gt_relevance_list)
        print(f"NDCG for the whole set for {model_type} (phase1): ", round(ndcg_sample_phase1*100,2))
        print(f"NDCG for the whole set for {model_type}(finetune)", round(ndcg_sample_finetune*100,2))

        ndcg_write_path = self._get_ndcg_path(self.model_preds_root,
                                              model_type, phase="sparse",
                                              subset_type=self.subset_type)
        print(f"Saving as {ndcg_write_path}")
        self.write_list_to_file(ndcg_write_path, ndcg_list_phase1)

        ndcg_write_path = self._get_ndcg_path(self.model_preds_root,
                                              model_type, phase="finetune",
                                              subset_type=self.subset_type)
        print(f"Saving as {ndcg_write_path}")
        self.write_list_to_file(ndcg_write_path, ndcg_list_finetune)

    def get_ndcg_value_wrapper(self, ranks: List,
                               gt_relevance : List):
        """

        :param ranks: list
        :param gt_relevance: list
        :return:
        """
        # (batch_size, num_options)
        ranks = torch.tensor(ranks).float()
        gt_relevance = torch.tensor(gt_relevance).float()
        # If individual sample, we need to add 0-dim
        if len(ranks.size()) == 1:
            ranks = ranks.unsqueeze(0)
            gt_relevance = gt_relevance.unsqueeze(0)
        self.ndcg.observe(ranks, gt_relevance)
        value = self.ndcg.retrieve(reset=True)["ndcg"]
        return value

    def get_both_phase_ranks(self, model_type,
                             ranks_phase1_file="ranks_val_12_crowdsourced.json",
                             ranks_finetune_file="ranks_val_best_ndcg_crowdsourced.json"):
        """

        :param model_type:
        :param ranks_phase1_file:
        :param ranks_finetune_file:
        :return:
        """
        model_rank_phase1_path = Path(self.model_preds_root, model_type, ranks_phase1_file)
        model_rank_finetune_path = Path(self.model_preds_root, model_type, ranks_finetune_file)
        # list of dic -> dict_keys(['image_id', 'round_id', 'ranks'])
        ranks_phase1_json = self.json_load(model_rank_phase1_path)
        ranks_finetune_json = self.json_load(model_rank_finetune_path)
        return ranks_phase1_json, ranks_finetune_json

    def subset_val_ranks_with_dense_annotation(self, ranks_json):
        """
        this is because val ranks consists of 10 turns
        :param ranks_json:
        :param top_k:
        :return:
        """
        rank_dense_list = []

        # gt_indices_list = []  # 0-indexed
        # gt_relevance_list = []
        # dialogs = self.data_val['data']['dialogs']

        for i in range(len(self.annotations_json)):
            # They will be in same order by image_id
            round_id = self.annotations_json[i]['round_id'] - 1  # 0-indexing
            index_for_ranks_json = i * 10 + round_id  # for each image: 10
            assert ranks_json[index_for_ranks_json]['round_id'] == round_id + 1  # Check with 1-indexing
            assert ranks_json[index_for_ranks_json]['image_id'] == self.annotations_json[i]['image_id']
            rank_dense_list.append(ranks_json[index_for_ranks_json])

            # ranks = ranks_json[index_for_ranks_json]['ranks']
            # image_id = ranks_json[index_for_ranks_json]['image_id']
            # gt_relevance = self.annotations_json[i]['gt_relevance']
            # # To actually have the indices_list and relevance list before hand
            # assert image_id == dialogs[i]['image_id']
            # gt_index = dialogs[i]['dialog'][round_id]['gt_index']
            # # round_id already 0-index gt_index also 0-indexed
            # gt_ans_relevance = gt_relevance[gt_index]
            # gt_indices_list.append(gt_index)
            # gt_relevance_list.append(gt_ans_relevance)
            # # We need to find rank of (gt_index + 1) - coz ranks are 1-indexed
            # pred_index = ranks.index(gt_index + 1)  # maintaining 0-index

        return rank_dense_list


    @staticmethod
    def write_list_to_file(filepath, write_list) -> None:
        with open(filepath, 'w') as file_handler:
            for item in write_list:
                file_handler.write("{}\n".format(item))
        # outfile.write("\n".join(itemlist))
        return

    @staticmethod
    def json_load(file_path):
        with open(file_path, "r") as fb:
            data = json.load(fb)
        return data

    @staticmethod
    def get_model_type_list(model_preds_root) -> List:
        model_folder_list = [os.path.basename(x) for x in glob.glob(os.path.join(model_preds_root, '*'))]
        print("Total models in folder:", len(model_folder_list))
        return model_folder_list

    @staticmethod
    def _get_ndcg_path(model_root: str,
                       model_type: str,
                       phase: str,
                       subset_type: str = '_crowdsourced',
                       ext: str = 'txt') -> str:
        json_path = f"{model_root}/{model_type}/ndcg_{phase}_{model_type}{subset_type}.{ext}"
        return json_path



def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-v", "--path_val_data",
    #     default="data/crowdsourced/visdial_1.0_val_crowdsourced.json",
    #     help="Path to data directory with json dialogues."
    # )
    parser.add_argument(
        "-a", "--dense_annotations_jsonpath",
        default="data/crowdsourced/visdial_1.0_val_dense_annotations_crowdsourced.json",
        help="Path to data directory with annotations."
    )
    parser.add_argument(
        "-m", "--model_preds_root", default="models/visdialconv/",
        help="Path to root model dir."
    )
    parser.add_argument(
        "-s", "--subset_type", default="",
        help="Subset type."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    ranker = NDCGForRanks(config)
    model_folder_list = ranker.get_model_type_list(config.model_preds_root)
    # print(f"Tying for {model_folder_list[0]}")
    for model_type in model_folder_list:
        ranker.extract_ndcg(model_type)
