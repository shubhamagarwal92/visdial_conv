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
# from random import choice, sample
import random

class DenseAnnotationsVisDial:
    """
    We will be subsetting actual dataset based on subset image ids
    """
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir
        # Read image ids for which we want to subset
        self.annotation_type = config.annotation_type

        # GT_1 means keep the gt as 1 only
        if self.annotation_type == "gt_1":
            self.data_type="train"  # We are fixing annotations for train only
            data_path = self._get_json_path(data_dir=self.data_dir,
                                            data_type=self.data_type)
            data_json = self.json_load(data_path)
            # print(data_json.keys()) # ['version', 'split', 'data']
            # print(data_json['data'].keys()) # ['questions', 'dialogs', 'answers']
            questions = data_json['data']['questions']  # All questions
            answers = data_json['data']['answers']  # All answers
            dialogs = data_json['data']['dialogs']
            self.dialogs = self.convert_list_of_json_to_json_dic(dialogs)
            print("Data loaded")
        else:
            self.sample_values = (0, 0.5, 1)

    def dump_new_dense_annotations(self, data_type="train"):
        # follow augment_dense_annotations.py
        new_dense_annotation_list = []
        dense_annotations_jsonpath = self._dense_json_path(self.data_dir, data_type=data_type)
        annotations_json = self.json_load(dense_annotations_jsonpath)

        if self.annotation_type == "gt_1":
            for ann_indx in range(len(annotations_json)):

                annotation_line = annotations_json[ann_indx]
                image_id = annotation_line["image_id"]
                round_id = annotation_line["round_id"]
                relevance = annotation_line["relevance"]

                # round id -> 1-index; gt_index -> 0-index
                dialog = self.dialogs[image_id]
                # print(dialog.keys())
                assert dialog['image_id'] == image_id
                gt_index = dialog['dialog'][round_id -1]['gt_index']
                relevance[gt_index] = 1  # maintaining 0-index

                assert type(relevance) == list
                annotation_line["relevance"] = relevance
                new_dense_annotation_list.append(annotation_line)
                # Assert the values are changed
                assert new_dense_annotation_list[ann_indx]["relevance"][gt_index] == 1

        else:
            # Generate random dense annotations
            # print(annotations_json[0].keys())
            # List of json
            # relevance - train; gt_relevance - val
            # list of dic -> dict_keys(['image_id', 'round_id', 'relevance'])
            for ann_indx in range(len(annotations_json)):

                annotation_line = annotations_json[ann_indx]
                image_id = annotation_line["image_id"]
                round_id = annotation_line["round_id"]
                relevance = annotation_line["relevance"]
                assert type(relevance) == list
                new_relevance = []
                for option_indx in range(len(relevance)):
                    # Uniform distribution
                    new_relevance.append(random.choice(self.sample_values))

                annotation_line["relevance"] = new_relevance
                new_dense_annotation_list.append(annotation_line)


        print("Total new annotations are: {} ", len(new_dense_annotation_list))
        self.new_dense_annotation_path = self.new_dense_json_path(config.save_data_dir,
                                                                  data_type=data_type,
                                                                  file_type_name=self.annotation_type)
        print(f"Saving new annotations in {self.new_dense_annotation_path}")
        self.json_dump(self.new_dense_annotation_path, out_json=new_dense_annotation_list)

        return


    @staticmethod
    def convert_list_of_json_to_json_dic(data_json):
        image_index_dic = {}
        for i in range(len(data_json)):
            image_index_dic[data_json[i]["image_id"]] = data_json[i]
        return image_index_dic


    @staticmethod
    def json_load(file_path):
        with open(file_path, "r") as fb:
            data = json.load(fb)
        return data

    @staticmethod
    def json_dump(file_path, out_json):
        with open(file_path, 'w') as outfile:
            json.dump(out_json, outfile)
        return

    @staticmethod
    def _get_json_path(data_dir: str,
                       data_type: str,
                       split: str = '1.0') -> str:
        """
        Call as _get_json_path(data_dir=data_dir, data_type=data_type)
        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        json_path = f"{data_dir}/visdial_{split}_{data_type}.json"
        return json_path

    @staticmethod
    def _dense_json_path(data_dir: str,
                         data_type: str,
                         split: str = '1.0') -> str:
        json_path = f"{data_dir}/visdial_{split}_{data_type}_dense_annotations.json"
        return json_path

    @staticmethod
    def new_dense_json_path(data_dir: str,
                            data_type: str,
                            file_type_name: str,
                            split: str = '1.0') -> str:
        json_path = f"{data_dir}/visdial_{split}_{data_type}_dense_annotations_{file_type_name}.json"
        return json_path

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="/scratch/shubham/visdial2019/data/",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-s", "--save_data_dir", default="/scratch/shubham/visdial2019/data/",
        help="Path to save data."
    )
    parser.add_argument("-t", "--annotation_type", choices=["gt_1", "uniform"],
                        default="gt_1",
                        help="Kind of new dense. GT_1 means keep the gt as 1 only")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    extractor = DenseAnnotationsVisDial(config)
    extractor.dump_new_dense_annotations("train")
