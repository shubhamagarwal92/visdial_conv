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
from pathlib import Path
import torch
from scipy.stats import f_oneway
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests

class SignificanceTest:
    def __init__(self, config):
        super().__init__()
        self.model_preds_root = config.model_preds_root
        self.models_list = self.get_model_type_list(self.model_preds_root)
        self.phase_list = ["sparse", "finetune"]
        self.subset_type = config.subset_type

    def get_test(self):
        """
        :param model_type: for which we want to extract
        :return:
        """

        print(f"Calculating sign test for subset: {self.subset_type}")

        # Load all models predictions for each phase
        for phase in self.phase_list:
            phase_all_models_ndcg_list = []  # ndcg list
            model_type_list = []  # Name of the model

            for model_type in self.models_list:
                ndcg_path = self._get_ndcg_path(self.model_preds_root, model_type,
                                                phase=phase, subset_type=self.subset_type)
                ndcg_list = self.read_file_as_list(ndcg_path)
                print(f"Total samples {model_type}: {len(ndcg_list)}")
                phase_all_models_ndcg_list.append(ndcg_list)
                model_type_list.append(model_type)

            # We form combinations of all indices
            index_models_list = list(range(len(model_type_list)))
            # pairwise
            combination_set = combinations(index_models_list, 2)

            for combination_indices in combination_set:
                model1_preds = phase_all_models_ndcg_list[combination_indices[0]]
                model2_preds = phase_all_models_ndcg_list[combination_indices[1]]
                model1_name = model_type_list[combination_indices[0]]
                model2_name = model_type_list[combination_indices[1]]

                stat, p = mannwhitneyu(model1_preds, model2_preds)
                print(f'Mannwhitneyu - For phase: {phase} - models: {model1_name} vs'
                      f' {model2_name} : stat={stat:.4f}, p={p:.4f}')

                stat, p = wilcoxon(model1_preds, model2_preds)
                print(f'Wilcoxon - For phase: {phase} - models: {model1_name} vs'
                      f' {model2_name} : stat={stat:.4f}, p={p:.4f}')

            # Checking for equivalence of *args
            # stat, p = f_oneway(phase_all_models_ndcg_list[0],
            # phase_all_models_ndcg_list[1], phase_all_models_ndcg_list[2],
            # phase_all_models_ndcg_list[3])
            # stat, p = f_oneway(*phase_all_models_ndcg_list)
            # stat, p = mannwhitneyu(*phase_all_models_ndcg_list)
            # stat, p = wilcoxon(*phase_all_models_ndcg_list)
            stat, p = kruskal(*phase_all_models_ndcg_list)
            print(f'Kruskal - For phase: {phase}: stat={stat:.4f}, p={p:.4f}')

            bonferroni_correction = multipletests(p, method='bonferroni')
            # print(bonferroni_correction)
            # (reject, pvals_corrected, alphacSidak, alphacBonf)
            action = str(bonferroni_correction[0][0])  # np array
            new_p_value = bonferroni_correction[1][0]
            print(f'Kruskal - bonferroni - For phase: {phase}: p={new_p_value:.4f}, '
                  f'action: {str(action)}')


            stat, p = friedmanchisquare(*phase_all_models_ndcg_list)
            print(f'Friedmanchisquare - For phase: {phase}: stat={stat:.4f}, p={p:.4f}')
            # To test on R
            # df=pd.DataFrame(data=phase_all_models_ndcg_list)
            # df = df.transpose()
            # df.columns=model_type_list
            # all_model_results = self._get_all_ndcg_path(self.model_preds_root, phase)
            # print(f"Saving df to: {all_model_results}")
            # df.to_csv(all_model_results)


    @staticmethod
    def read_file_as_list(filepath):
        with open(filepath, 'r') as file_handler:
            return_list = file_handler.readlines()
            return_list = [float(line.strip()) for line in return_list]
        return return_list


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

    @staticmethod
    def _get_all_ndcg_path(model_root: str,
                       phase: str,
                       ext: str = 'txt') -> str:
        json_path = f"{model_root}//ndcg_{phase}_all.{ext}"
        return json_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_preds_root", default="models/visdialconv/",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-s", "--subset_type", default="",
        help="Subset type."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    sign_test = SignificanceTest(config)
    sign_test.get_test()

