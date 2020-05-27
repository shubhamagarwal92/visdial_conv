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

class SubsetImageIDs:
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir
        self.all_image_ids = []

    def extract_image_ids(self):
        # We dont care in which split the images are from vispro,
        # we will be subsetting from visdial dataset
        for data_type in ['train', 'val', 'test']:
            data_list = self._get_vispro_data(self.data_dir, data_type=data_type)
            # Weird formatting
            # print(data_list[0].keys())
            for indx in range(len(data_list)):
                self.all_image_ids.append(self.image_id_from_path(json.loads(data_list[indx])['image_file']))

        vispro_image_id_path = self._vispro_img_ids_path(data_dir=self.save_data_dir)
        self.write_list_to_file(vispro_image_id_path, self.all_image_ids)

    @staticmethod
    def write_list_to_file(filepath, write_list):
        with open(filepath, 'w') as file_handler:
            for item in write_list:
                file_handler.write("{}\n".format(item))
        # outfile.write("\n".join(itemlist))
        return

    @staticmethod
    def inspect_dir(data_dir):
        folder_list = glob.glob(os.path.join(data_dir, '*'))
        print(folder_list)
        print("Total files: ", len(folder_list))
        return folder_list

    @staticmethod
    def image_id_from_path(image_path):
        """Given a path to an image, return its id.

        Parameters
        ----------
        image_path : str
            Path to image, e.g.: coco_train2014/COCO_train2014/000000123456.jpg
            img_name = "VisualDialog_val2018_000000254080.jpg"
        Returns
        -------
        int
            Corresponding image id (123456)
        """

        return int(image_path.split("/")[-1][-16:-4])

    @staticmethod
    def _get_vispro_data(data_dir: str,
                         data_type: str) -> List:
        _path = f"{data_dir}/{data_type}.vispro.1.1.jsonlines"
        with open(_path, 'r') as file_handle:
            data_list = file_handle.readlines()
        return data_list

    @staticmethod
    def _get_json_path(data_dir: str,
                       data_type: str,
                       split: str = '1.0') -> str:
        """
        Call as
        get_json_path(data_dir=data_dir, data_type=data_type)

        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        json_path = f"{data_dir}/visdial_{split}_{data_type}.json"
        return json_path

    @staticmethod
    def _vispro_img_ids_path(data_dir: str,
                             ext: str = 'txt') -> str:
        """
        Call as
        _vispro_img_ids_path(data_dir=data_dir, ext)

        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        file_path = f"{data_dir}/vispro_image_ids.{ext}"
        return file_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="data/",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-s", "--save_data_dir", default="data/",
        help="Path to data directory with json dialogues."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    extractor = SubsetImageIDs(config)
    extractor.extract_image_ids()
