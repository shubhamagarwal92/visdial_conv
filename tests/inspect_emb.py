import h5py
import numpy as np

def inspect(features_hdfpath):
    features_hdf = h5py.File(features_hdfpath, "r")
    keys = list(features_hdf.keys())
    print(keys) # ['hist_embeddings', 'image_id']
    attrs = list(features_hdf.attrs.keys())
    print(attrs)

    # hist_embeddings = features_hdf['hist_embeddings']
    # image_id = features_hdf['image_id']
    # print(hist_embeddings.shape)
    # print(image_id.shape)

    ques_embeddings = features_hdf['ques_embeddings']
    ans_embeddings = features_hdf['ans_embeddings']
    captions_embeddings = features_hdf['captions_embeddings']

    print(len(features_hdf["ques_embeddings"]))
    print(len(features_hdf["ans_embeddings"]))

    print(ques_embeddings.shape)
    print(ans_embeddings.shape)
    print(captions_embeddings.shape)

    # train
    # (376082, 20, 768)
    # (337527, 20, 768)

    # Val
    # (45237, 20, 768)
    # (34821, 20, 768)

    # test
    # (45237, 20, 768)
    # (34821, 20, 768)

    split = features_hdf.attrs['split']
    print(split)


def inspect_index(features_hdfpath):
    features_hdf = h5py.File(features_hdfpath, "r")
    keys = list(features_hdf.keys())
    print(keys) # ['hist_embeddings', 'image_id']
    attrs = list(features_hdf.attrs.keys())
    print(attrs)
    # ques_embeddings = features_hdf['ques_embeddings']
    # ans_embeddings = features_hdf['ans_embeddings']
    # captions_embeddings = features_hdf['captions_embeddings']

    index = np.random.randint(16, size=(10))
    index = np.random.randint(low = 0, high = 3, size = 10)
    print(index)

    index_list = [1, 2, 4]
    index_array = np.array(index_list)

    print(index_list)
    print(index_array)

    # https://github.com/h5py/h5py/issues/992
    # https://stackoverflow.com/questions/38761878/indexing-a-large-3d-hdf5-dataset-for-subsetting-based-on-2d-condition

    # indexed_q_embeddings = features_hdf['ques_embeddings'][index]
    indexed_q_embeddings = features_hdf['ques_embeddings'][index_list]
    indexed_q_embeddings = features_hdf['ques_embeddings'][index_array]
    print(indexed_q_embeddings.shape)


if __name__ == "__main__":
    data_dir = "../../data/bert"
    hist_emb_file_name = "visdial_1.0_train_bert_concat_true_sep_False_emb.h5"
    qa_emb_file_name = "visdial_1.0_test_emb.h5"


    qa_emb_file_name = "visdial_1.0_train_qac_bert_emb.h5"

    emb_file_name = qa_emb_file_name
    # emb_file_name = hist_emb_file_name
    features_hdfpath = "{}/{}".format(data_dir, emb_file_name)
    # inspect(features_hdfpath)
    inspect_index(features_hdfpath)
