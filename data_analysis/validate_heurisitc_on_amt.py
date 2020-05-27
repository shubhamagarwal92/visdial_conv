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

        if config.create_stats_dic:
            self.nlp = spacy.load('en_core_web_sm')
            # self.coref_model = pretrained.neural_coreference_resolution_lee_2017()
            # Constituency parser for getting ellipsis
            self.const_parser = pretrained.span_based_constituency_parsing_with_elmo_joshi_2018()
            self.heuristic_root_cp = ["S", "SQ", "SBARQ", "SINV"]

        self.non_agreement_images = [128889, 525402, 270717, 562489, 204402, 431203, 566063, 391624, 34733, 357189, 194408, 290919, 179535, 321201, 434695, 422086, 389671, 219511, 288804, 135635, 506795, 565265, 76825, 113894, 569745, 454202, 648, 188270, 62886, 67435, 301586, 274049, 139477, 372849, 418086, 520026, 358062, 24025, 531916, 129793, 165015, 491809, 58794, 77166, 177397, 130307, 469621, 249431, 133036, 542075, 240603, 330121, 526149, 472987, 456541, 498784, 216234, 30786, 51931, 512332, 31706, 220643, 39771, 170070, 386874, 100528, 231110, 154609, 139962, 145204, 567307, 174821, 468028, 114981, 284112, 406280, 528160, 185561, 313049, 306269, 325335, 332510, 546153, 389417, 489182, 174023, 95844, 116883, 452505, 53772, 87395, 290313, 225029, 439314, 494256, 84393, 118025, 417343, 57931, 190947, 509900, 60776, 235054, 307886, 544849, 70689, 51184, 225737, 127730, 227148, 240892, 167060, 49132, 300446, 467899, 123729, 402930, 567184, 185894, 333125, 323557, 1872, 573045, 353753, 142963, 268723, 327133, 161055, 185565, 574189, 29737, 99643, 295627, 485732, 546554, 286929, 175345, 223379, 146821, 358981, 427711, 212259, 347890, 297528, 83797, 369360, 538790, 219444, 101666, 201002]
        self.hist_info_images = [257366, 425477, 191097, 552399, 12468, 458949, 109735, 311793, 437200, 355853, 98849, 57743, 83289, 488471, 446567, 196905, 308846, 328336, 289233, 52156, 366462, 511748, 457675, 518811, 413085, 432039, 531270, 430580, 293582, 544148, 80366, 179366, 150236, 400960, 10424, 451398, 498340, 268914, 384171, 172461, 387266, 214227, 555578, 181772, 149373, 251385, 407878, 574545, 544827, 120559, 19299, 73638, 496822, 204195, 97073, 209447, 53433, 403234, 524006, 178300, 376460, 570468, 292100, 227006, 170315, 456824, 525726, 179064, 98879, 558975, 193521, 377823, 449230, 44468, 573552, 288308, 237956, 69538, 250654, 439842, 146314, 458818, 122826, 33976, 322815, 239030, 209271, 560666, 361734, 225491, 27366, 29060, 191186, 394073, 120870, 580183, 111013]

    @staticmethod
    def read_amt_results(file_path):
        amt_results = []
        with open(file_path, "r") as fb:
            for line in fb:
                amt_results.append(json.loads(line))
        return amt_results

    def return_agreement_images(self, amt_results):
        # We will filter out the hit ids
        hit_ids = []
        question_list = []
        option_list = []
        image_list = []
        for hit in amt_results:
            hit_results = hit['output']  # actual output per each hit
            hit_id = hit['hit_id']
            if hit_id in hit_ids:
                continue
            else:
                hit_ids.append(hit_id)
                for indx in range(len(hit_results)):
                    image_id = hit_results[indx]['image_id']
                    # if image_id not in self.non_agreement_images:
                        # image_list.append(hit_results[indx]['image_id'])
                        # question_list.append(hit_results[indx]['question'])
                        # option_list.append(hit_results[indx]['option'])

                    image_list.append(hit_results[indx]['image_id'])
                    question_list.append(hit_results[indx]['question'])
                    option_list.append(hit_results[indx]['option'])
        return image_list, question_list, option_list


    def create_stats_dic(self, data_type: str):
        self.results_jsonpath = self._get_json_path(self.data_dir, data_type)
        print(f"Reading json {self.results_jsonpath}")

        amt_results = self.read_amt_results(self.results_jsonpath)
        print("Total AMT hits: ", len(amt_results))

        image_list, question_list, option_list = self.return_agreement_images(amt_results)
        print(len(question_list))


        stats_dic = {
            "pronouns_ques": [],
            "non_pleonastic_pronouns_ques": [],
            "ellipsis_ques": [],
        }

        # All list have same length
        for i in tqdm(range(len(question_list))):
            question = question_list[i]
            # Get question related
            ellipsis_ques = self._get_ellipsis(question)
            non_pleonastic_pronouns_ques, pronouns_ques = self._get_pronouns(question)

            stats_dic["non_pleonastic_pronouns_ques"].append(non_pleonastic_pronouns_ques)
            stats_dic["pronouns_ques"].append(pronouns_ques)
            stats_dic["ellipsis_ques"].append(ellipsis_ques)

        stats_dic["image_id"] = image_list
        stats_dic["option"] = option_list


        # Save the dic as pkl
        pkl_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type="amt")
        self.pickle_dump(pkl_file_path, stats_dic)
        return

    def get_analysis(self, data_type: str):
        """
        Assumes stats dic has been created. just do analysis here
        :param data_type:
        :return:
        """

        self.results_jsonpath = self._get_json_path(self.data_dir, data_type)
        print(f"Reading json {self.results_jsonpath}")
        amt_results = self.read_amt_results(self.results_jsonpath)
        print("Total AMT hits: ", len(amt_results))
        image_list, question_list, option_list = self.return_agreement_images(amt_results)


        pkl_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type="amt")
        print("Reading stats dic data from: ", pkl_file_path)
        stats_dic = self.pickle_load(pkl_file_path)

        # SA new
        stats_dic["image_id"] = image_list
        stats_dic["option"] = option_list  # TODO : this may be corrupted

        per_turn_keys = ['non_pleonastic_pronouns_ques', 'pronouns_ques', 'ellipsis_ques']

        non_agreement_indices = []

        for key in per_turn_keys:
            list_key = stats_dic[key]
            assert len(list_key) == len(image_list)
            # TP -- Cases with pronoun and marked hist info
            # FP -- Cases with pronoun and marked no hist info
            # FN -- Cases with no pronoun and marked hist info
            # TN -- Cases with no pronoun and marked no hist info
            #
            # Precision -- (TP)/(TP+ FP)
            # Recall -- (TP)/(TP+ FN)

            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for indx in range(len(image_list)):
                image_id = image_list[indx]

                if image_id not in self.non_agreement_images:

                    if (list_key[indx] !=0 and image_id in self.hist_info_images):
                        tp += 1
                    if (list_key[indx] !=0 and image_id not in self.hist_info_images):
                        fp += 1
                    if (list_key[indx] ==0 and image_id in self.hist_info_images):
                        fn += 1
                    if (list_key[indx] ==0 and image_id not in self.hist_info_images):
                        tn += 1

                    # tp += 1 if (list_key !=0 and image_id in self.hist_info_images) else 0
                    # fp += 1 if list_key !=0 and image_id not in self.hist_info_images
                    # fn += list_key ==0 and image_id in self.hist_info_images
                    # tn += list_key ==0 and image_id not in self.hist_info_images

                else:
                    non_agreement_indices.append(indx)

            print("True positive: ", tp)
            print("False positive: ", fp)
            print("False negative: ", fn)
            print("True negative: ", tn)

            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            fscore = 2 * precision*recall/ (precision + recall)

            print(f"Precession for {key}: {precision}")
            print(f"Recall for {key}: {recall}")
            print(f"F-score for {key}: {fscore}")


        per_turn_stats = dict((k, stats_dic[k]) for k in per_turn_keys if k in stats_dic)


        per_turn_stats_df = pd.DataFrame.from_dict(per_turn_stats)
        per_turn_stats_df = per_turn_stats_df.loc[~per_turn_stats_df.index.isin(non_agreement_indices)]

        per_turn_stats_df_describe = per_turn_stats_df.describe()


        write_file_path = self._save_file_path(save_data_dir=self.save_data_dir, data_type=data_type, ext="xlsx")
        self.write_excel_df(write_file_path,
                            [per_turn_stats_df_describe],
                            ['turn_level'])


        write_file_path = self._save_file_path(save_data_dir=self.save_data_dir,
                                               data_type=data_type, file_type_name="percent",
                                               ext="xlsx")

        writer = pd.ExcelWriter(write_file_path, engine='xlsxwriter')

        self.write_value_counts_df_excel_offset(write_file_path=write_file_path,
                                                sheet_name='Turn level',
                                                key_list=per_turn_keys,
                                                write_df=per_turn_stats_df,
                                                writer=writer)
        writer.save()
        print("Non-pleonastic pronouns")
        print(get_column_stats(per_turn_stats_df, 'non_pleonastic_pronouns_ques'))
        print("All pronouns")
        print(get_column_stats(per_turn_stats_df, 'pronouns_ques'))
        print("Cases of ellipsis")
        print(get_column_stats(per_turn_stats_df, 'ellipsis_ques'))

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
        json_path = f"{data_dir}/results_amt_batch_1_5.txt"
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
        inspector.create_stats_dic(data_type="amt")

    inspector.get_analysis(data_type="amt")
