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

class SubsetVisDial:
    """
    We will be subsetting actual dataset based on subset image ids
    """
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir
        # Read image ids for which we want to subset
        self.image_id_list_path = config.image_id_list_path
        self.subset_image_ids = self.read_file_as_list(self.image_id_list_path)
        self.subset_type = config.subset_type

    def dump_subset_data(self, data_type="val"):
        data_path = self._get_json_path(data_dir=self.data_dir,
                                        data_type=data_type)
        data_json = self.json_load(data_path)

        # print(data_json.keys()) # ['version', 'split', 'data']
        # print(data_json['data'].keys()) # ['questions', 'dialogs', 'answers']

        questions = data_json['data']['questions']  # All questions
        answers = data_json['data']['answers']  # All answers
        dialogs = data_json['data']['dialogs']

        # Subset of dialogs by image ids
        subset_dialogs = []

        print(self.subset_image_ids)
        for dialog in dialogs:
            # print(dialog)
            if dialog["image_id"] in self.subset_image_ids:
                subset_dialogs.append(dialog)

        print("Total image ids for which we want to subset: {}".format(len(self.subset_image_ids)))
        print("Total dialogs in the subset are: {} ".format(len(subset_dialogs)))

        data = {
            "questions": questions,
            "answers": answers,
            "dialogs": subset_dialogs
        }

        out_json = {
            "version": data_json['version'],
            "split": data_json['split'],
            "data": data
        }

        save_file_path = self._subset_file_path(save_data_dir=config.save_data_dir,
                                                data_type=data_type,
                                                file_type_name=self.subset_type)
        self.json_dump(save_file_path, out_json=out_json)

    def dump_subset_dense_annotations(self, data_type):
        # follow augment_dense_annotations.py
        self.new_dense_annotation_list = []
        dense_annotations_jsonpath = self._dense_json_path(self.data_dir, data_type=data_type)
        annotations_json = self.json_load(dense_annotations_jsonpath)

        # List of json
        for ann_indx in range(len(annotations_json)):
            image_id = annotations_json[ann_indx]["image_id"]
            if image_id in self.subset_image_ids:
                self.new_dense_annotation_list.append(annotations_json[ann_indx])

        print("Total annotations in the subset are: {} ", len(self.new_dense_annotation_list))
        self.new_dense_annotation_path = self.new_dense_json_path(config.save_data_dir,
                                                                  data_type=data_type,
                                                                  file_type_name=self.subset_type)
        self.json_dump(self.new_dense_annotation_path, out_json=self.new_dense_annotation_list)

        return

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
    def read_file_as_list(filepath):
        with open(filepath, 'r') as file_handler:
            return_list = file_handler.readlines()
            return_list = [int(line.strip()) for line in return_list]
        return return_list


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
        """
        Call as _dense_json_path(data_dir=data_dir, data_type=data_type)
        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        json_path = f"{data_dir}/visdial_{split}_{data_type}_dense_annotations.json"
        return json_path

    @staticmethod
    def new_dense_json_path(data_dir: str,
                            data_type: str,
                            file_type_name: str,
                            split: str = '1.0') -> str:
        """
        Call as _dense_json_path(data_dir=data_dir, data_type=data_type)
        :param data_dir:
        :param data_type:
        :param file_type_name:
        :param split:
        :return:
        """
        json_path = f"{data_dir}/visdial_{split}_{data_type}_dense_annotations_{file_type_name}.json"
        return json_path

    @staticmethod
    def _subset_file_path(save_data_dir: str,
                        data_type: str,
                        file_type_name: str = 'vispro',
                        split: str = '1.0',
                        ext: str = 'json') -> str:
        """
        Call as _subset_file_path(data_dir=data_dir, data_type=data_type)
        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        file_path = f"{save_data_dir}/visdial_{split}_{data_type}_{file_type_name}.{ext}"
        return file_path

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="data/",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-s", "--save_data_dir", default="data/",
        help="Path to save data."
    )
    parser.add_argument("-i", "--image_id_list_path",
                        help="Path to image id list")
    parser.add_argument("-t", "--subset_type",
                        help="Kind of subset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    extractor = SubsetVisDial(config)
    extractor.dump_subset_data("val")
    extractor.dump_subset_dense_annotations("val")
