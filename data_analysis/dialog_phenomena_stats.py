import argparse
from tqdm import tqdm
import numpy as np
from allennlp import pretrained
import json
import typing
import copy
import pandas as pd
from typing import List, Dict, Union, Any
import spacy
from collections import defaultdict, Counter
import pickle as pkl


# Pandas helper functions
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


def subset_df_from_val(df, col_name, value):
    subset_df = df.loc[df[col_name] == value]
    return subset_df



class InspectDialogPhenomena(object):
    """
    Adapated from original readers

    A simple reader for VisDial v1.0 dialog data. The json file must have the
    same structure as mentioned on ``https://visualdialog.org/data``.

    Parameters
    ----------
    config : argparse object
    """

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir
        # Heuristic weather conditions
        self.weather_list = ['rainy', 'sunny', 'daytime', 'day', 'night']
        self.difficult_pronouns = ["other", "it"]
        self.nlp = spacy.load('en_core_web_sm')
        # self.coref_model = pretrained.neural_coreference_resolution_lee_2017()
        # Constituency parser for getting ellipsis
        self.const_parser = pretrained.span_based_constituency_parsing_with_elmo_joshi_2018()
        self.heuristic_root_cp = ["S", "SQ", "SBARQ", "SINV"]

    def create_stats_dic(self, data_type: str):
        data_path = self._get_json_path(self.data_dir, data_type)
        print(f"Reading json {data_path}")
        json_data = json.load(open(data_path))
        # Questions
        questions = json_data['data']['questions']
        print("Total questions: ", len(questions))
        # Answers
        answers = json_data['data']['answers']
        print("Total answers: ", len(answers))
        # Dialogs
        dialogs = json_data['data']['dialogs']

        stats_dic = {
            "image_id": [],
            "pronouns_ques_dialog": [],
            "pronouns_ans_dialog": [],
            "pronouns_ques": [],
            "pronouns_ans": [],
            "pronouns_dialog": [],
            "non_pleonastic_pronouns_ques_dialog": [],
            "non_pleonastic_pronouns_ans_dialog": [],
            "non_pleonastic_pronouns_ques": [],
            "non_pleonastic_pronouns_ans": [],
            "non_pleonastic_pronouns_dialog": [],
            "ellipsis_ques_dialog": [],
            "ellipsis_ans_dialog": [],
            "ellipsis_ques": [],
            "ellipsis_ans": [],
            "ellipsis_dialog": [],
            "pronouns_caption": [],
            "non_pleonastic_pronouns_caption": [],
            "ellipsis_caption": [],
            "ques_list_dialog": [],
            "ans_list_dialog": [],
            "caption_list_dialog": []
        }

        for dialog_id in tqdm(range(len(dialogs))):
            image_id = dialogs[dialog_id]['image_id']
            dialog_for_image = dialogs[dialog_id]['dialog']
            caption = dialogs[dialog_id]['caption']

            # Get caption related stats
            ellipsis_caption = self._get_ellipsis(caption)
            non_pleonastic_pronouns_caption, pronouns_caption = self._get_pronouns(caption)

            # Default values for each dialog
            pronouns_ques_dialog = 0
            pronouns_ans_dialog = 0
            pronouns_dialog = 0
            non_pleonastic_pronouns_ques_dialog = 0
            non_pleonastic_pronouns_ans_dialog = 0
            non_pleonastic_pronouns_dialog = 0
            ellipsis_ques_dialog = 0
            ellipsis_ans_dialog = 0
            ellipsis_dialog = 0

            ques_list_dialog = []
            ans_list_dialog = []
            for round_id in range(len(dialog_for_image)):
                question = questions[dialog_for_image[round_id]['question']]
                answer = answers[dialog_for_image[round_id]['answer']]
                # Append question and answer per dialog list
                ques_list_dialog.append(question)
                ans_list_dialog.append(answer)

                # Get question related
                ellipsis_ques = self._get_ellipsis(question)
                non_pleonastic_pronouns_ques, pronouns_ques = self._get_pronouns(question)

                # Get answer related
                ellipsis_ans = self._get_ellipsis(answer)
                non_pleonastic_pronouns_ans, pronouns_ans = self._get_pronouns(answer)

                # Save per ques/ans stats
                stats_dic["non_pleonastic_pronouns_ques"].append(non_pleonastic_pronouns_ques)
                stats_dic["non_pleonastic_pronouns_ans"].append(non_pleonastic_pronouns_ans)
                stats_dic["pronouns_ques"].append(pronouns_ques)
                stats_dic["pronouns_ans"].append(pronouns_ans)
                stats_dic["ellipsis_ques"].append(ellipsis_ques)
                stats_dic["ellipsis_ans"].append(ellipsis_ans)

                # Accumulating on dialog level
                pronouns_ques_dialog += pronouns_ques
                pronouns_ans_dialog += pronouns_ans
                pronouns_dialog += (pronouns_ques + pronouns_ans)
                non_pleonastic_pronouns_ques_dialog += non_pleonastic_pronouns_ques
                non_pleonastic_pronouns_ans_dialog += non_pleonastic_pronouns_ans
                non_pleonastic_pronouns_dialog += (non_pleonastic_pronouns_ques + non_pleonastic_pronouns_ans)
                ellipsis_ques_dialog += ellipsis_ques
                ellipsis_ans_dialog += ellipsis_ans
                ellipsis_dialog += (ellipsis_ques + ellipsis_ans)

            # Caption related
            stats_dic["image_id"].append(image_id)
            stats_dic["ellipsis_caption"].append(ellipsis_caption)
            stats_dic["non_pleonastic_pronouns_caption"].append(non_pleonastic_pronouns_caption)
            stats_dic["pronouns_caption"].append(pronouns_caption)

            # Save per dialog stats
            stats_dic["pronouns_ques_dialog"].append(pronouns_ques_dialog)
            stats_dic["pronouns_ans_dialog"].append(pronouns_ans_dialog)
            stats_dic["pronouns_dialog"].append(pronouns_dialog)
            stats_dic["non_pleonastic_pronouns_ques_dialog"].append(non_pleonastic_pronouns_ques_dialog)
            stats_dic["non_pleonastic_pronouns_ans_dialog"].append(non_pleonastic_pronouns_ans_dialog)
            stats_dic["non_pleonastic_pronouns_dialog"].append(non_pleonastic_pronouns_dialog)
            stats_dic["ellipsis_ques_dialog"].append(ellipsis_ques_dialog)
            stats_dic["ellipsis_ans_dialog"].append(ellipsis_ans_dialog)
            stats_dic["ellipsis_dialog"].append(ellipsis_dialog)

            # Save raw ques and ans list per dialog
            stats_dic["ques_list_dialog"].append(ques_list_dialog)
            stats_dic["ans_list_dialog"].append(ans_list_dialog)
            stats_dic["caption_list_dialog"].append(caption)

        # Save the dic as pkl
        pkl_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type=data_type)
        self.pickle_dump(pkl_file_path, stats_dic)
        return

    def get_analysis(self, data_type: str):
        """
        Assumes stats dic has been created. just do analysis here
        :param data_type:
        :return:
        """
        pkl_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type=data_type)
        print("Reading stats dic data from: ", pkl_file_path)
        stats_dic = self.pickle_load(pkl_file_path)

        per_turn_keys = ['non_pleonastic_pronouns_ques', 'non_pleonastic_pronouns_ans',
                         'pronouns_ques', 'pronouns_ans', 'ellipsis_ques', 'ellipsis_ans']
        per_turn_stats = dict((k, stats_dic[k]) for k in per_turn_keys if k in stats_dic)

        per_dialog_keys = ['pronouns_ques_dialog', 'pronouns_ans_dialog',
                           'pronouns_dialog', 'non_pleonastic_pronouns_ques_dialog',
                           'non_pleonastic_pronouns_ans_dialog', 'ellipsis_ques_dialog',
                           'ellipsis_ans_dialog', 'ellipsis_dialog', 'ellipsis_caption',
                           'non_pleonastic_pronouns_caption', 'pronouns_caption']
        per_dialog_stats = dict((k, stats_dic[k]) for k in per_dialog_keys if k in stats_dic)


        per_turn_stats_df = pd.DataFrame.from_dict(per_turn_stats)
        per_turn_stats_df_describe = per_turn_stats_df.describe()

        per_dialog_stats_df = pd.DataFrame.from_dict(per_dialog_stats)
        per_dialog_stats_df_describe = per_dialog_stats_df.describe()

        write_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type=data_type, ext="xlsx")
        self.write_excel_df(write_file_path,
                            [per_turn_stats_df_describe, per_dialog_stats_df_describe],
                            ['turn_level', 'dialog_level'])

        # subset_df = subset_df_from_val(per_dialog_stats_df, 'non_pleonastic_pronouns_ques_dialog', 12)
        # print(subset_df)

        write_file_path = self._save_file_path(save_data_dir=self.save_data_dir,
                                               data_type=data_type, file_type_name="percent",
                                               ext="xlsx")

        writer = pd.ExcelWriter(write_file_path, engine='xlsxwriter')
        self.write_value_counts_df_excel_offset(write_file_path=write_file_path,
                                                sheet_name='Dialog level',
                                                key_list=per_dialog_keys,
                                                write_df=per_dialog_stats_df,
                                                writer=writer)

        self.write_value_counts_df_excel_offset(write_file_path=write_file_path,
                                                sheet_name='Turn level',
                                                key_list=per_turn_keys,
                                                write_df=per_turn_stats_df,
                                                writer=writer)
        writer.save()
        print(get_column_stats(per_dialog_stats_df, 'non_pleonastic_pronouns_ques_dialog'))
        print(get_column_stats(per_turn_stats_df, 'non_pleonastic_pronouns_ques'))

    @staticmethod
    def write_value_counts_df_excel_offset(write_file_path: str,
                                           sheet_name: str,
                                           key_list: List,
                                           write_df,
                                           writer = None,
                                           start_pos: int = 0,
                                           offset: int = 3,
                                           engine: str = 'xlsxwriter'):
        """
        TODO: SA better way to handle this
        https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas
        :param write_file_path:
        :param sheet_name:
        :param key_list:
        :param write_df:
        :param writer:
        :param start_pos:
        :param offset:
        :param engine:
        :return:
        """
        if writer is None:
            writer = pd.ExcelWriter(write_file_path, engine=engine)
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet
        for key in key_list:
            column_stats_df = get_column_stats(write_df, key)
            worksheet.write_string(start_pos, 0, key)
            column_stats_df.to_excel(writer, sheet_name=sheet_name, startrow=start_pos + 1, startcol=0)
            start_pos = start_pos + offset + len(column_stats_df.index)
        # writer.save()
        return

    def _get_pronouns(self, text):
        doc = self.nlp(text)
        non_pleonastic_pronoun = 0
        pronouns = 0
        for token in doc:
            if token.pos_ == "PRON":
                pronouns += 1
                non_pleonastic_pronoun += 1
            # Heuristic 1
            # To remove pleonastic_pronoun such as "is it sunny" as case of pronoun
            if token.text == "it" and any(weather_element in text for weather_element in self.weather_list):
                non_pleonastic_pronoun -= 1

        # Heuristic 2:
        # what about the other - other should be marked as pronoun here
        if "other" in text:
            pronouns += 1
            non_pleonastic_pronoun += 1

        # if any(weather_element in text for weather_element in self.weather_list):
        #     pleonastic_pronoun = 0
        return non_pleonastic_pronoun, pronouns

    def _get_ellipsis(self, text):
        ellipsis = 0
        const_results = self.const_parser.predict(text)
        root = const_results['trees'].replace('(','').split(" ")[0]
        if root not in self.heuristic_root_cp:
            ellipsis = 1

        return ellipsis

    @staticmethod
    def pickle_dump(save_file_path: str, pickle_obj: Any):
        """
        Dump object into pkl format
        :param save_file_path:
        :param pickle_obj:
        :return:
        """
        with open(save_file_path, 'wb') as outfile:
            pkl.dump(pickle_obj, outfile)
        return

    @staticmethod
    def pickle_load(file_path: str):
        """
        Load a pickle object
        :param file_path:
        :return:
        """
        with open(file_path, 'rb') as infile:
            pickle_obj = pkl.load(infile)
        return pickle_obj

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
    def _save_file_path(save_data_dir: str,
                        data_type: str,
                        split: str = '1.0',
                        file_type_name: str = 'analysis',
                        ext: str = 'pkl') -> str:
        """
        Call as
        get_json_path(data_dir=data_dir, data_type=data_type)

        :param data_dir:
        :param data_type:
        :param split:
        :return:
        """
        file_path = f"{save_data_dir}/visdial_{split}_{data_type}_{file_type_name}.{ext}"
        return file_path

    @staticmethod
    def write_excel_df(save_file_path: str,
                       df_list: List, sheet_name_list: List):
        """
        Save a list of df in different sheets in one excel file
        :param save_file_path:
        :param df_list:
        :param sheet_name_list:
        :return:
        """
        writer = pd.ExcelWriter(save_file_path, engine='xlsxwriter')
        # Write each dataframe to a different worksheet
        assert len(df_list) == len(sheet_name_list)
        for index in range(len(df_list)):
            df_list[index].to_excel(writer, sheet_name=sheet_name_list[index])
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", default="data/",
        help="Path to data directory."
    )
    parser.add_argument(
        "-s", "--save_data_dir", default="data/",
        help="Path to save data directory."
    )
    parser.add_argument(
        "-c", "--create_stats_dic", action="store_true",
        help="If we want to create stats dic or just do analysis."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    inspector = InspectDialogPhenomena(config=config)
    if config.create_stats_dic:
        inspector.create_stats_dic(data_type="val")
        inspector.create_stats_dic(data_type="train")

    inspector.get_analysis(data_type="val")
    inspector.get_analysis(data_type="train")


    # inspector.get_analysis(data_type="test")
