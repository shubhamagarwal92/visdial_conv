import pandas as pd
import json


def count_unique(df, col_name):
    """ Count unique values in a df column """
    count = df[col_name].nunique()
    return count


def get_unique_column_values(df,col_name):
    """ Returns unique values """
    return df[col_name].unique()


def get_column_stats(df, column_name, to_dict = False):
    if to_dict:
        return df[column_name].value_counts().to_dict()
    else:
        # return df[column_name].value_counts()
        c = df[column_name].value_counts(dropna=False)
        p = df[column_name].value_counts(dropna=False, normalize=True)*100
        m = pd.concat([c,p], axis=1, keys=['counts', '%'])
        return m


def get_pandas_percentile(df, key):
    df[key].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    return


def print_analysis(_list, key="relevance",  _type="relevance"):
    df = pd.DataFrame(_list)
    df.columns = [key]
    print("\n-----------------------------------")
    print("Total unique {} responses".format(_type))
    print(count_unique(df,key))
    print("\n-----------------------------------")
    print("Stats for {} responses".format(_type))
    print(get_column_stats(df,key))
    print("\n-----------------------------------")
    print("Number of {} responses".format(_type))
    print(df[key].describe())
    return



def json_load(file_path):
    with open(file_path, "r") as fb:
        data = json.load(fb)
    return data


def convert_list_json_dic(ranks_json):
    image_ranks_dic = {}
    for i in range(len(ranks_json)):
        image_ranks_dic[ranks_json[i]["image_id"]] = ranks_json[i]
    return image_ranks_dic


def main(dialogs_jsonpath, dense_annotations_jsonpath, data_type = "val"):
    dialogs_reader = json_load(dialogs_jsonpath)
    annotations_json = json_load(dense_annotations_jsonpath)

    dialogs = dialogs_reader["data"]["dialogs"]  # list of dialogs
    annotations_json = convert_list_json_dic(annotations_json)
    # print(annotations_json)

    gt_relevance_list = []

    for dial_index in range(len(dialogs)):
        image_id = dialogs[dial_index]["image_id"]
        dialog_for_image = dialogs[dial_index]["dialog"]

        # This condition for train set
        if image_id in annotations_json:
            dense_annotations = annotations_json[image_id]
            gt_round_id = dense_annotations["round_id"] -1   # Converting to 0 index
            gt_image_id = dense_annotations["image_id"]

            if data_type == "train":
                # print(dense_annotations.keys())
                gt_relevance = dense_annotations["relevance"]
            else:
                gt_relevance = dense_annotations["gt_relevance"]

            _dialog = dialog_for_image[gt_round_id]
            gt_index = _dialog["gt_index"]

            _gt_relevance = gt_relevance[gt_index]
            gt_relevance_list.append(_gt_relevance)

            # print("Length of gt relevance:", len(gt_relevance))

    print(len(gt_relevance_list))
    df = pd.DataFrame(gt_relevance_list)
    df.columns = ['relevance']
    print(df)
    print_analysis(gt_relevance_list)


if __name__ == '__main__':
    dialogs_jsonpath = "data/visdial_1.0_val.json"
    dense_annotations_jsonpath = "data/visdial_1.0_val_dense_annotations.json"

    main(dialogs_jsonpath, dense_annotations_jsonpath)


    dialogs_jsonpath = "data/visdial_1.0_train.json"
    dense_annotations_jsonpath = "data/visdial_1.0_train_dense_annotations.json"

    main(dialogs_jsonpath, dense_annotations_jsonpath, data_type="train")
