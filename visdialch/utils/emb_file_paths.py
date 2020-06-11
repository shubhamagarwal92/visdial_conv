import os

KEYS = ["ques_embeddings", "ans_embeddings", "captions_embeddings",
        "hist_embeddings", "image_ids", "opts_embeddings"]
KEYS_TO_REMOVE = ["img_ids", "hist"]


def get_json_path(data_dir: str,
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


# SA: should be used by all embeddings related files
def get_emb_dir_file_path(data_dir: str,
                          emb_type: str) -> str:
    """
    Embedding directory file path
    :param data_dir:
    :param emb_type: as "bert"
    :return:
    """
    dir_file_path = f"{data_dir}/{emb_type}"
    # if not os.path.exists(dir_file_path):
    #     os.makedirs(dir_file_path)
    return dir_file_path


def get_per_dial_emb_file_path(data_dir: str, image_id: int,
                               ext: str = 'npz') -> str:
    """
    Call as get_npz_file_path(data_dir=data_dir, image_id=image_id)
    :param data_dir:
    :param image_id:
    :param ext:
    :return:
    """
    file_path = f"{data_dir}/{str(image_id)}.{ext}"
    return file_path


# SA: bert_data_processing/dump_visdial_bert_hist.py
def get_hist_embeddings_file_path(data_dir: str,
                                  data_type: str,
                                  concat: bool,
                                  emb_type: str,
                                  split: str = '1.0',
                                  ext: str = "h5") -> str:
    """
    Call as
    get_hist_embeddings_file_path(data_dir=data_dir, data_type=data_type, concat=concat, emb_type=emb_type)


    :param data_dir:
    :param data_type:
    :param concat:
    :param emb_type:
    :param split:
    :param ext:
    :return:
    """
    """hist_concat_true_emb.h5"""
    file_path = f"{data_dir}/visdial_{split}_{data_type}_hist_concat_{str(concat)}_{emb_type}_emb.{ext}"
    return file_path


## SA:
def get_qa_embeddings_file_path(data_dir: str,
                                data_type: str,
                                emb_type: str,
                                split: str = '1.0',
                                ext: str = 'h5') -> str:
    """

    Call as
    get_qa_embeddings_file_path(data_dir=data_dir, data_type=data_type, emb_type=emb_type)

    :param data_dir:
    :param data_type:
    :param emb_type:
    :param split:
    :param ext:
    :return:
    """
    """ qac_bert_emb.h5"""

    """ Note: this also contains caption embeddings. Embeddings of all possible qac"""

    # file_path = (f"{data_dir}/{emb_type}/visdial_{split}_{data_type}_qac_{emb_type}_emb.{ext}")
    file_path = (f"{data_dir}/visdial_{split}_{data_type}_qac_{emb_type}_emb.{ext}")
    return file_path


# Not specific to any embedding file path
# SA: bert_data_processing/dump_visdial_hist.py
def get_indices_file_path(data_dir: str,
                          data_type: str,
                          split: str = '1.0',
                          ext: str = 'h5') -> str:
    """

    Call as
    get_indices_file_path(data_dir=data_dir, data_type=data_type)


    :param data_dir:
    :param data_type:
    :param split:
    :param ext:
    :return:
    """

    """Get file path to save indices"""
    file_path = f"{data_dir}/visdial_{split}_{data_type}_indices.{ext}"
    return file_path


# SA: new
def get_dial_qa_embeddings_file_path(data_dir: str,
                                     data_type: str,
                                     emb_type: str,
                                     split: str = '1.0',
                                     ext: str = 'h5') -> str:
    """
    QA embeddings in dialog format

    :param data_dir:
    :param data_type:
    :param concat:
    :param emb_type:
    :param split:
    :param ext:
    :return:
    """
    file_path = f"{data_dir}/visdial_{split}_{data_type}_{emb_type}_qa_dial.{ext}"
    return file_path


# SA: new
def get_combined_embeddings_file_path(data_dir: str,
                                      data_type: str,
                                      concat: bool,
                                      emb_type: str,
                                      split: str = '1.0',
                                      ext: str = 'h5') -> str:
    file_path = f"{data_dir}/visdial_{split}_{data_type}_{emb_type}_concat_hist_{str(concat)}_all.{ext}"
    return file_path


# SA called by data/extract_text_data.py
def get_qac_raw_file_path(self, data_dir: str, data_type: str,
                          split: str = '1.0',
                          ext: str = 'pkl') -> str:
    file_path = f"{data_dir}/visdial_{split}_{data_type}_raw_text.{ext}"
    return file_path


# SA called by data/extract_text_data.py
def get_raw_txt_file_path(data_dir: str, data_type: str,
                          text_type: str, split: str = '1.0') -> str:
    """
        Call as

        get_raw_txt_file_path(data_dir=data_dir, data_type=data_type, text_type="questions")

    :param data_dir:
    :param data_type:
    :param split:
    :param text_type:
    :return:
    """
    file_path = f"{data_dir}/visdial_{split}_{data_type}_{text_type}.txt"

    return file_path


def get_raw_hist_file_path(data_dir: str,
                           data_type: str,
                           split: str = '1.0',
                           concat: bool = False,
                           ext: str = 'pkl') -> str:
    """
    Call as

    get_raw_hist_file_path(data_dir=data_dir, data_type=data_type)

    :param data_dir:
    :param split:
    :param concat:
    :param ext:
    :return:
    """
    """Get file path to save hist"""
    file_path = f"{data_dir}/visdial_{split}_{data_type}_raw_hist_concat_{str(concat)}.{ext}"
    return file_path


def get_flatten_csv_file_path(data_dir: str,
                              data_type: str,
                              concat: bool = False,
                              ques_ans_eos: bool = False,
                              split: str = '1.0',
                              ext: str = 'csv') -> str:
    """
    Call as

    get_raw_hist_file_path(data_dir=data_dir, data_type=data_type)

    :param data_dir:
    :param split:
    :param concat:
    :param ext:
    :return:
    """
    """Get file path to save hist"""
    file_path = f"{data_dir}/visdial_{split}_{data_type}" \
        f"_flatten_data_hist_concat_{str(concat)}_qa_sep_" \
        f"{str(ques_ans_eos)}.{ext}"

    return file_path


def get_dense_json_path(data_dir: str,
                        data_type: str,
                        split: str = '1.0') -> str:
    """
    Call as

    get_dense_json_path(data_dir=data_dir, data_type=data_type)

    :param data_dir:
    :param data_type:
    :param split:
    :return:
    """

    json_path = f"{data_dir}/visdial_{split}_{data_type}_dense_annotations.json"
    return json_path

# Deprecated
## SA: currrent embeddings file path
# def get_embeddings_file_path(data_dir: str = "data",
#                            type: str = "train",
#                            split: str = '1.0') -> str:
#     file_path = "{}/visdial_{}_{}_emb.h5".format(data_dir, split, type)
#     return file_path
