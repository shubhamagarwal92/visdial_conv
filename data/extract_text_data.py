import argparse
from tqdm import tqdm
from typing import Any, Dict
import json
import pickle
import numpy as np
import pandas as pd

def count_unique(df, col_name):
    """ Count unique values in a df column """
    count = df[col_name].nunique()
    return count


def get_unique_column_values(df,col_name):
    """ Returns unique values """
    return df[col_name].unique()


def get_column_stats(df,column_name,to_dict = False):
    if to_dict:
        return df[column_name].value_counts().to_dict()
    else:
        # return df[column_name].value_counts()
        c = df[column_name].value_counts(dropna=False)
        p = df[column_name].value_counts(dropna=False, normalize=True)*100
        m = pd.concat([c,p], axis=1, keys=['counts', '%'])
        return m


def get_pandas_percentile(df):
    df['words'].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    return


class SubsetExtractor:
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir

    def extract_text(self, type: str):
        data_path = self._get_json_path(type)
        print(f"Reading json {data_path}")
        data_json = json.load(open(data_path))

        # print(data_json.keys()) # ['version', 'split', 'data']
        # print(data_json['data'].keys()) # ['questions', 'dialogs', 'answers']

        # SA: todo subset only relevant questions and answers for speed up
        questions = data_json['data']['questions']  # All questions
        answers = data_json['data']['answers']  # All answers
        dialogs = data_json['data']['dialogs']

        captions = []
        image_ids = []
        for dialog in dialogs:
            # print(dialog.keys()) # dict_keys(['image_id', 'dialog', 'caption'])
            captions.append(dialog['caption'])
            image_ids.append(dialog['image_id'])

            # Iterate over dialog as
            # dialog_obj = dialog["dialog"]
            # for turn in dialog_obj:
            #     captions.append(turn['caption'])

        print("Number of dialogs: {}".format(len(dialogs)))
        print("Number of questions: {}".format(len(questions)))
        print("Number of answers: {}".format(len(answers)))
        print("Number of captions: {}".format(len(captions)))
        save_file_path = self._save_file_path(type)

        text_data = {
            "questions": questions,
            "answers": answers,
            "captions": captions
        }

        self.print_analysis(captions)
        self.print_analysis(questions, "questions")
        self.print_analysis(answers, "answers")

        with open(save_file_path, 'wb') as outfile:
            pickle.dump(text_data, outfile)

        captions_file_path = self._save_txt_file_path(dataset_type=type, list_type="captions")

        with open(captions_file_path, 'w') as outfile:
            for item in captions:
                outfile.write("{}\n".format(item.encode('utf-8')))
        print("Captions written")

        questions_file_path = self._save_txt_file_path(dataset_type=type, list_type="questions")
        with open(questions_file_path, 'w') as outfile:
            for item in questions:
                outfile.write("{}\n".format(item.encode('utf-8')))

        answers_file_path = self._save_txt_file_path(dataset_type=type, list_type="answers")
        with open(answers_file_path, 'w') as outfile:
            for item in answers:
                outfile.write("{}\n".format(item.encode('utf-8')))

        image_ids_file_path = self._save_txt_file_path(dataset_type=type, list_type="image_ids")
        with open(image_ids_file_path, 'w') as outfile:
            for item in image_ids:
                outfile.write("{}\n".format(item))


        # SA: working example of data loading
        # Test load
        # print("Trying to load now")
        # with open(save_file_path, 'rb') as outfile:
        #     pickle_obj = pickle.load(outfile)
        #
        # questions = pickle_obj["questions"]
        # answers = pickle_obj["answers"]
        # captions = pickle_obj["captions"]
        # print("Number of questions: {}".format(len(questions)))
        # print("Number of answers: {}".format(len(answers)))
        # print("Number of captions: {}".format(len(captions)))

    def print_analysis(self, _list, _type="caption"):
        df = pd.DataFrame(_list)
        df.columns = ['text']
        df['words'] = df['text'].str.split().str.len()
        print("\n-----------------------------------")
        print("Total unique {} responses".format(_type))
        print(count_unique(df,'text'))
        print("\n-----------------------------------")
        print("Stats for {} responses".format(_type))
        print(get_column_stats(df,'text'))
        print("\n-----------------------------------")
        print("Number of words in text for {} responses".format(_type))
        print(df['words'].describe())
        return

    def get_stats_list(self, _list, _type="caption"):
        """SA: Deprecated for now."""
        lengths = map(len,_list)
        print(lengths)
        average = float(sum(lengths))/float(len(_list))
        print("Max length of {} list {}".format(_type, max(lengths)))
        print("Avg length of {} list {}".format(_type, average))
        p = np.percentile(lengths, 50)
        print("Median of {} list {}".format(_type, average))
        return

    def _get_json_path(self, type: str,
                       split: str = '1.0') -> str:
        json_path = "{}/visdial_{}_{}.json".format(self.data_dir, split, type)
        return json_path

    def _save_file_path(self, type: str,
                        split: str = '1.0') -> str:
        file_path = "{}/visdial_{}_{}_raw_text.pkl".format(self.save_data_dir, split, type)
        return file_path

    def _save_txt_file_path(self, dataset_type: str = "train",
                            split: str = '1.0',
                            list_type: str = "captions") -> str:

        file_path = f"{self.save_data_dir}/visdial_{split}_{dataset_type}_{list_type}.txt"

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
    extractor = SubsetExtractor(config)
    extractor.extract_text("val")
    extractor.extract_text("train")
    extractor.extract_text("test")
