import pandas as pd
import argparse
import json
import numpy as np
from collections import defaultdict


class AugmentDenseAnnotations:
    def __init__(self, dialogs_jsonpath: str,
                 dense_annotations_jsonpath: str,
                 save_dense_annotations_jsonpath: str,
                 data_type: str = "train"):
        self.dialogs_jsonpath = dialogs_jsonpath
        self.dense_annotations_jsonpath = dense_annotations_jsonpath
        self.data_type = data_type
        self.save_dense_annotations_jsonpath = save_dense_annotations_jsonpath
        # We will save dialogs as well
        self.dialogs = {}
        self.correlation_dic = {}
        self.flat_correlation_dic = defaultdict(dict)
        self.new_dense_annotation_list = []

    @staticmethod
    def json_load(file_path):
        with open(file_path, "r") as fb:
            data = json.load(fb)
        return data

    @staticmethod
    def convert_list_json_dic(data_json):
        image_index_dic = {}
        for i in range(len(data_json)):
            image_index_dic[data_json[i]["image_id"]] = data_json[i]
        return image_index_dic

    def get_correlation_dic(self):
        dialogs_reader = self.json_load(self.dialogs_jsonpath)
        annotations_json = self.json_load(self.dense_annotations_jsonpath)

        #  Convert list of dialogs to dic of jsons
        # Saving raw dialogs. used later
        self.dialogs = dialogs_reader["data"]["dialogs"]  # list of dialogs

        dialogs = self.convert_list_json_dic(self.dialogs)

        print("Total dialogs (indexed by image_id): ", len(dialogs))
        print("Total annotations: ", len(annotations_json))
        print("Keys for annotation json: ", annotations_json[0].keys())
        print("Data type: ", self.data_type)

        # Create the dic for each annotation example
        for ann_indx in range(len(annotations_json)):
            image_id = annotations_json[ann_indx]["image_id"]
            round_id = annotations_json[ann_indx]["round_id"]-1   # Converting to 0 index

            if self.data_type == "train":
                # To tackle gt_relevance
                gt_relevance = annotations_json[ann_indx]["relevance"]
            else:
                gt_relevance = annotations_json[ann_indx]["gt_relevance"]  # list of annotations

            dialog_for_image = dialogs[image_id]["dialog"]
            _dialog = dialog_for_image[round_id]
            gt_index = _dialog["gt_index"]
            ans_opts = _dialog["answer_options"]

            # Ans opts - actual options
            # gt_index - index in ans_opts
            #
            gt_ans = ans_opts[gt_index]
            self.correlation_dic[gt_ans] = {}
            for opt_indx in range(len(ans_opts)):
                if opt_indx == gt_index:
                    continue
                current_ans_opt = ans_opts[opt_indx]
                relevance = gt_relevance[opt_indx]

                # https://stackoverflow.com/questions/12905999/python-dict-how-to-create-key-or-append-an-element-to-key
                self.correlation_dic[gt_ans].setdefault(current_ans_opt, []).append(relevance)

        corr_ans_keys = self.correlation_dic.keys()
        print("Number of unique answers with other correlation answers: ", len(corr_ans_keys))
        #  Normalize the values

        for ans in corr_ans_keys:
            other_correlated_ans = self.correlation_dic[ans].keys()
            for other_ans in other_correlated_ans:
                list_dense_score = self.correlation_dic[ans][other_ans]
                normalize_value = np.mean(list_dense_score)
                self.flat_correlation_dic[ans][other_ans] = normalize_value
                self.flat_correlation_dic[other_ans][ans] = normalize_value

        all_keys = self.flat_correlation_dic.keys()
        print("All possible correlation: ", len(all_keys))

        #  Create new dense annotations

        num_dialogs_changed = 0
        num_examples_changed = 0
        num_values_changed = 0
        total_dialogs = len(self.dialogs)
        total_examples = 0
        for dial_index in range(total_dialogs):

            image_id = self.dialogs[dial_index]["image_id"]
            dialog_for_image = self.dialogs[dial_index]["dialog"]

            image_annotation_list = []
            for round_id in range(len(dialog_for_image)):
                total_examples += 1
                _dialog = dialog_for_image[round_id]
                gt_index = _dialog["gt_index"]
                answer_options = _dialog["answer_options"]
                relevance_list = [0.] * len(answer_options)
                relevance_list[gt_index] = 1.
                gt_ans = answer_options[gt_index]

                current_round_values_changed = 0.
                if gt_ans in self.flat_correlation_dic:
                    dic_dense_score = self.flat_correlation_dic[gt_ans]
                    for ans_indx in range(len(answer_options)):
                        current_ans = answer_options[ans_indx]
                        if current_ans in dic_dense_score:
                            normalize_value = dic_dense_score[current_ans]
                            if normalize_value > 0:
                                num_values_changed += 1
                                current_round_values_changed += 1
                                relevance_list[ans_indx] = normalize_value

                if current_round_values_changed>0:
                    num_examples_changed += 1

                new_dense_annotation = {
                    "image_id": image_id,
                    "round_id": round_id
                }
                if self.data_type == "train":
                    # To tackle gt_relevance
                    new_dense_annotation["relevance"] = relevance_list
                else:
                    new_dense_annotation["gt_relevance"] = relevance_list

                # self.new_dense_annotation_list.append(new_dense_annotation)
                image_annotation_list.append(new_dense_annotation)

            per_image_annotation = {
                "image_id": image_id,
                "dense_annotation": image_annotation_list
            }

            self.new_dense_annotation_list.append(per_image_annotation)

        print("Values changed: ", num_values_changed)
        print("Examples changed: ", num_examples_changed)
        print("Total examples: ", total_examples)
        print("% new dense annotations: ", num_examples_changed*100/total_examples)
        print("Length of new dense annotations: ", len(self.new_dense_annotation_list))

        with open(self.save_dense_annotations_jsonpath, 'w') as outfile:
            json.dump(self.new_dense_annotation_list, outfile)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dialogs_jsonpath", default="data/visdial_1.0_train.json",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-a", "--dense_annotations_jsonpath", default="data/visdial_1.0_train_dense_annotations.json",
        help="Path to data directory with json dialogues."
    )
    parser.add_argument(
        "-s", "--save_dense_annotations_jsonpath", default="data/visdial_1.0_train_augmented_dense_annotations.json",
        help="Path to data directory with json dialogues."
    )

    args = parser.parse_args()
    return args


def main(args):
    augment_class = AugmentDenseAnnotations(args.dialogs_jsonpath,
                                            args.dense_annotations_jsonpath,
                                            args.save_dense_annotations_jsonpath)
    augment_class.get_correlation_dic()


if __name__ == '__main__':
    args = parse_args()
    main(args)
