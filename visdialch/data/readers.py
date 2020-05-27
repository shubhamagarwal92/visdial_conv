"""
A Reader simply reads data from disk and returns it almost as is, based on
a "primary key", which for the case of VisDial v1.0 dataset, is the
``image_id``. Readers should be utilized by torch ``Dataset``s. Any type of
data pre-processing is not recommended in the reader, such as tokenizing words
to integers, embedding tokens, or passing an image through a pre-trained CNN.

Each reader must atleast implement three methods:
    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on ``image_id`` in VisDial v1.0
      dataset.
    - ``keys`` to return a list of possible ``image_id``s this Reader can
      provide data of.
"""

import copy
import json
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Set, Union

import h5py

# A bit slow, and just splits sentences to list of words, can be doable in
# `DialogsReader`.
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle as pkl

class DialogsReader(object):
    """
    A simple reader for VisDial v1.0 dialog data. The json file must have the
    same structure as mentioned on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dialogs_jsonpath : str
        Path to json file containing VisDial v1.0 train, val or test data.
    num_examples: int, optional (default = None)
        Process first ``num_examples`` from the split. Useful to speed up while
        debugging.
    """

    def __init__(
        self,
        dialogs_jsonpath: str,
        num_examples: Optional[int] = None,
        num_workers: int = 1,
        use_pretrained_emb: bool = False
    ):
        with open(dialogs_jsonpath, "r") as visdial_file:
            visdial_data = json.load(visdial_file)
            self._split = visdial_data["split"]
            # SA: pre-trained embeddings
            self.use_pretrained_emb = use_pretrained_emb
            # Maintain questions and answers as a dict instead of list because
            # they are referenced by index in dialogs. We drop elements from
            # these in "overfit" mode to save time (tokenization is slow).
            self.questions = {
                i: question for i, question in
                enumerate(visdial_data["data"]["questions"])
            }
            self.answers = {
                i: answer for i, answer in
                enumerate(visdial_data["data"]["answers"])
            }

            # Add empty question, answer - useful for padding dialog rounds
            # for test split.
            self.questions[-1] = ""
            self.answers[-1] = ""

            # ``image_id``` serves as key for all three dicts here.
            self.captions: Dict[int, Any] = {}
            self.dialogs: Dict[int, Any] = {}
            self.num_rounds: Dict[int, Any] = {}
            self.original_indices: Dict[int, Any] = {}

            all_dialogs = visdial_data["data"]["dialogs"]

            # Retain only first ``num_examples`` dialogs if specified.
            if num_examples is not None:
                all_dialogs = all_dialogs[:num_examples]

            index = 0
            for _dialog in all_dialogs:

                self.captions[_dialog["image_id"]] = _dialog["caption"]
                self.original_indices[_dialog["image_id"]] = index
                index += 1


                # Record original length of dialog, before padding.
                # 10 for train and val splits, 10 or less for test split.
                self.num_rounds[_dialog["image_id"]] = len(_dialog["dialog"])

                # Pad dialog at the end with empty question and answer pairs
                # (for test split).
                while len(_dialog["dialog"]) < 10:
                    _dialog["dialog"].append({"question": -1, "answer": -1})

                # Add empty answer (and answer options) if not provided
                # (for test split). We use "-1" as a key for empty questions
                # and answers.
                for i in range(len(_dialog["dialog"])):
                    if "answer" not in _dialog["dialog"][i]:
                        _dialog["dialog"][i]["answer"] = -1
                    if "answer_options" not in _dialog["dialog"][i]:
                        _dialog["dialog"][i]["answer_options"] = [-1] * 100

                self.dialogs[_dialog["image_id"]] = _dialog["dialog"]

            # If ``num_examples`` is specified, collect questions and answers
            # included in those examples, and drop the rest to save time while
            # tokenizing. Collecting these should be fast because num_examples
            # during debugging are generally small.
            if num_examples is not None:
                questions_included: Set[int] = set()
                answers_included: Set[int] = set()

                for _dialog in self.dialogs.values():
                    for _dialog_round in _dialog:
                        questions_included.add(_dialog_round["question"])
                        answers_included.add(_dialog_round["answer"])
                        for _answer_option in _dialog_round["answer_options"]:
                            answers_included.add(_answer_option)

                self.questions = {
                    i: self.questions[i] for i in questions_included
                }
                self.answers = {
                    i: self.answers[i] for i in answers_included
                }

            print(f"[{self._split}] Tokenizing questions...")
            _question_tuples = self.questions.items()
            _question_indices = [t[0] for t in _question_tuples]
            _questions = list(tqdm(map(word_tokenize, [t[1] for t in _question_tuples])))
            self.questions = {
                i: question + ["?"] for i, question in
                zip(_question_indices, _questions)
            }
            # Delete variables to free memory.
            del _question_tuples, _question_indices, _questions

            print(f"[{self._split}] Tokenizing answers...")
            _answer_tuples = self.answers.items()
            _answer_indices = [t[0] for t in _answer_tuples]
            _answers = list(tqdm(map(word_tokenize, [t[1] for t in _answer_tuples])))
            # SA: adding "." instead of "?"
            self.answers = {
                i: answer for i, answer in
                zip(_answer_indices, _answers)
            }
            del _answer_tuples, _answer_indices, _answers

            print(f"[{self._split}] Tokenizing captions...")
            # Convert dict to separate lists of image_ids and captions.
            _caption_tuples = self.captions.items()
            _image_ids = [t[0] for t in _caption_tuples]
            _captions = list(tqdm(map(word_tokenize, [t[1] for t in _caption_tuples])))
            # Convert tokenized captions back to a dict.
            self.captions = {i: c for i, c in zip(_image_ids, _captions)}

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        original_index = self.original_indices[image_id]
        dialog = copy.copy(self.dialogs[image_id])
        # SA: added dialog index here
        if self.use_pretrained_emb:
            # Copy the dialog before converting to raw tokens
            dialog_with_index = copy.deepcopy(self.dialogs[image_id])
        num_rounds = self.num_rounds[image_id]

        # Replace question and answer indices with actual word tokens.
        for i in range(len(dialog)):
            dialog[i]["question"] = self.questions[
                dialog[i]["question"]
            ]
            dialog[i]["answer"] = self.answers[
                dialog[i]["answer"]
            ]
            for j, answer_option in enumerate(
                dialog[i]["answer_options"]
            ):
                dialog[i]["answer_options"][j] = self.answers[
                    answer_option
                ]

        visdial_instance = {
            "image_id": image_id,
            "caption": caption_for_image,
            "dialog": dialog,
            "num_rounds": num_rounds}

        # SA: we need integers to access the embeddings from the h5 files
        if self.use_pretrained_emb:
            visdial_instance["dialog_with_index"] = dialog_with_index
            visdial_instance["original_index"] = original_index

        return visdial_instance

    def keys(self) -> List[int]:
        return list(self.dialogs.keys())

    @property
    def split(self):
        return self._split


class DenseAnnotationsReader(object):
    """
    A reader for dense annotations for val split. The json file must have the
    same structure as mentioned on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dense_annotations_jsonpath : str
        Path to a json file containing VisDial v1.0
    """

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [
                entry["image_id"] for entry in self._visdial_data
            ]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    # SA: adding these APIs to get image indices for finetuning
    @property
    def all_data(self):
        return self._visdial_data

    # @property
    # def meta_dic(self) -> List:
    #     return self._meta_dic

    @property
    def keys(self) -> List[int]:
        return self._image_ids


    @property
    def split(self):
        # always
        return "val"

class AugmentedDenseAnnotationsReader(object):
    """
    A reader for dense annotations for val split. The json file must have the
    same structure as mentioned on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dense_annotations_jsonpath : str
        Path to a json file containing VisDial v1.0
    """

    def __init__(self, dense_annotations_jsonpath: str,
                 split: str = "train"):
        self._split = split
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [
                entry["image_id"] for entry in self._visdial_data
            ]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        dial_image_annotation = self._visdial_data[index]["dense_annotation"]

        # dial_image_annotation = self._visdial_data[image_id]["dense_annotation"]
        dial_annotation_list = []
        for round in range(len(dial_image_annotation)):
            # if self.split == "train": Always train
            dial_annotation_list.append(dial_image_annotation[round]["relevance"])
        # keys: {"image_id", "round_id", "gt_relevance"}
        return_dic = {
            "augmented_gt_relevance": dial_annotation_list
        }
        return return_dic

    # SA: adding these APIs to get image indices for finetuning
    @property
    def all_data(self):
        return self._visdial_data

    # @property
    # def meta_dic(self) -> List:
    #     return self._meta_dic

    @property
    def keys(self) -> List[int]:
        return self._image_ids


    @property
    def split(self):
        # always
        return self._split


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical
    HDF file is expected to have a column named "image_id", and another column
    named "features".

    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details
    about HDF structure.

    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split
        image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            self._image_id_list = list(features_hdf["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_id_list)

    def __getitem__(self, image_id: int):
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    self.features[index] = image_id_features
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]

        return image_id_features

    def keys(self) -> List[int]:
        return self._image_id_list

    @property
    def split(self):
        return self._split


class TransformerEmbeddingsHdfReader(object):
    """
    Same format as ImageFeaturesHdfReader.


    """
    def __init__(self, embedding_path: str,
                 in_memory: bool = False):
        self.embedding_path = embedding_path
        self._in_memory = in_memory

        with h5py.File(self.embedding_path, "r") as embedding_hdf:
            self._split = embedding_hdf.attrs["split"]
            self._image_id_list = list(embedding_hdf["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.ques_embeddings = [None] * len(self._image_id_list)
            self.hist_embeddings = [None] * len(self._image_id_list)
            self.opts_embeddings = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_ids)

    ## SA: todo check the return typing
    def __getitem__(self, image_id: int): # -> Dict[str, Union[int, List]]:
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            ## SA: check by ques_embeddings only. if it is there..all would be there!
            if self.ques_embeddings[index] is not None:
                ques_embeddings = self.ques_embeddings[index]
                hist_embeddings = self.hist_embeddings[index]
                opts_embeddings = self.opts_embeddings[index]
            else:
                with h5py.File(self.embedding_path, "r") as features_hdf:
                    ques_embeddings = features_hdf["ques_embeddings"][index]
                    hist_embeddings = features_hdf["hist_embeddings"][index]
                    opts_embeddings = features_hdf["opts_embeddings"][index]
                    self.ques_embeddings[index] = ques_embeddings
                    self.hist_embeddings[index] = hist_embeddings
                    self.opts_embeddings[index] = opts_embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.embedding_path, "r") as features_hdf:

                ques_embeddings = features_hdf["ques_embeddings"][index]
                hist_embeddings = features_hdf["hist_embeddings"][index]
                opts_embeddings = features_hdf["opts_embeddings"][index]

        embeddings = {"ques_embeddings": ques_embeddings,
                      "hist_embeddings": hist_embeddings,
                      "opts_embeddings": opts_embeddings}

        return embeddings

    def keys(self) -> List[int]:
        return self._image_id_list

    @property
    def split(self):
        # always
        return self._split


class QuesEmbeddingsHdfReader(object):
    """
    Same format as ImageFeaturesHdfReader.


    """
    def __init__(self, qa_emb_file_path: str,
                 in_memory: bool = False):
        """

        :param qa_emb_file_path: QA file path
        :param q_len: Number of questions
        :param a_len: Number of answers
        :param in_memory:
        """

        self.qa_emb_file_path = qa_emb_file_path
        self._in_memory = in_memory
        ## SA: trying to load everything
        if self._in_memory:
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self.ques_embeddings = qa_embedding_hdf["ques_embeddings"][:]
                print("All embedding loaded for questions: ", len(self.ques_embeddings))
                self.q_len = len(self.ques_embeddings)
        else:
            # SA:
            print("Loading the file only. Not reading in memory")
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self._split = qa_embedding_hdf.attrs["split"]

                # SA: todo check if we can do len or shape
                self.q_len = len(qa_embedding_hdf["ques_embeddings"])
                # "features" is List[np.ndarray] if the dataset is loaded in-memory
                # If not loaded in memory, then list of None.
                self.ques_embeddings = [None] * self.q_len

    # @todo What should be the length??
    def __len__(self):
        return len(self.q_len)

    ## SA: todo check the return typing
    def __getitem__(self,
                    q_index: int): # -> Dict[str, Union[int, List]]:
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            ## SA: check by ques_embeddings only. if it is there..all would be there!
            if self.ques_embeddings[q_index] is not None:
                ques_embeddings = self.ques_embeddings[q_index]
            else:
                with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                    ques_embeddings = features_hdf["ques_embeddings"][q_index]
                    # Store in memory
                    self.ques_embeddings[q_index] = ques_embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                ques_embeddings = features_hdf["ques_embeddings"][q_index]

        # embeddings = {"ques_embeddings": ques_embeddings,
        #               "ans_embeddings": ans_embeddings}

        return ques_embeddings

    # @todo: what should be here? -- index is actually the key
    def keys(self) -> List[int]:
        return list(range(self.q_len))

    @property
    def split(self):
        # always
        return self._split


class AnswerEmbeddingsHdfReader(object):
    """
    Same format as ImageFeaturesHdfReader.


    """
    def __init__(self, qa_emb_file_path: str,
                 in_memory: bool = False):
        """

        :param qa_emb_file_path: QA file path
        :param q_len: Number of questions
        :param a_len: Number of answers
        :param in_memory:
        """

        self.qa_emb_file_path = qa_emb_file_path
        self._in_memory = in_memory

        ## SA: trying to load everything
        if self._in_memory:
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self.ans_embeddings = qa_embedding_hdf["ans_embeddings"][:]
                print("All embedding loaded for answers", len(self.ans_embeddings))
                self.a_len = len(self.ans_embeddings)
        else:
            print("Loading the file only. Not reading in memory")
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self._split = qa_embedding_hdf.attrs["split"]
                # SA: todo check if we can do len or shape
                self.a_len = len(qa_embedding_hdf["ans_embeddings"])
                # "features" is List[np.ndarray] if the dataset is loaded in-memory
                # If not loaded in memory, then list of None.
                self.ans_embeddings = [None] * self.a_len

    # @todo What should be the length??
    def __len__(self):
        return len(self.a_len)


    ## SA: todo check the return typing
    def __getitem__(self,
                    a_index: int): # -> Dict[str, Union[int, List]]:
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            ## SA: check by ques_embeddings only. if it is there..all would be there!
            if self.ans_embeddings[a_index] is not None:
                ans_embeddings = self.ans_embeddings[a_index]
            else:
                with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                    ans_embeddings = features_hdf["ans_embeddings"][a_index]
                    # Store in memory
                    self.ans_embeddings[a_index] = ans_embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                ans_embeddings = features_hdf["ans_embeddings"][a_index]

        # embeddings = {"ques_embeddings": ques_embeddings,
        #               "ans_embeddings": ans_embeddings}

        return ans_embeddings

    # @todo: what should be here?
    def keys(self) -> List[int]:
        return list(range(self.a_len))

    @property
    def split(self):
        # always
        return self._split

# SA: Index here should refer to the actual dialog index while we are indexing by image id
class CaptionEmbeddingsHdfReader(object):
    """
    Same format as ImageFeaturesHdfReader.

    """
    def __init__(self, qa_emb_file_path: str,
                 in_memory: bool = False):
        """

        :param qa_emb_file_path: QA file path
        :param q_len: Number of questions
        :param a_len: Number of answers
        :param in_memory:
        """

        self.qa_emb_file_path = qa_emb_file_path
        self._in_memory = in_memory

        ## SA: trying to load everything
        if self._in_memory:
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self.captions_embeddings = qa_embedding_hdf["captions_embeddings"][:]
                print("All embedding loaded for answers", len(self.captions_embeddings))
                self.cap_len = len(self.captions_embeddings)
        else:
            print("Loading the file only. Not reading in memory")
            with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
                self._split = qa_embedding_hdf.attrs["split"]
                # SA: todo check if we can do len or shape
                self.cap_len = len(qa_embedding_hdf["captions_embeddings"])
                # "features" is List[np.ndarray] if the dataset is loaded in-memory
                # If not loaded in memory, then list of None.
                self.captions_embeddings = [None] * self.cap_len

        # with h5py.File(self.qa_emb_file_path, "r") as qa_embedding_hdf:
        #     self._split = qa_embedding_hdf.attrs["split"]
        #     # SA: todo check if we can do len or shape
        #     self.cap_len = len(qa_embedding_hdf["captions_embeddings"])
        #     # "features" is List[np.ndarray] if the dataset is loaded in-memory
        #     # If not loaded in memory, then list of None.
        #     self.caption_embeddings = [None] * self.cap_len

    def __len__(self):
        return len(self.cap_len)

    def __getitem__(self,
                    index: int):  # -> Dict[str, Union[int, List]]:
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            ## SA: check by ques_embeddings only. if it is there..all would be there!
            if self.captions_embeddings[index] is not None:
                captions_embeddings = self.captions_embeddings[index]
            else:
                with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                    captions_embeddings = features_hdf["captions_embeddings"][index]
                    # Store in memory
                    self.captions_embeddings[index] = captions_embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.qa_emb_file_path, "r") as features_hdf:
                captions_embeddings = features_hdf["captions_embeddings"][index]

        return captions_embeddings

    # @todo: what should be here?
    def keys(self) -> List[int]:
        return list(range(self.cap_len))

    @property
    def split(self):
        # always
        return self._split


class HistEmbeddingsHdfReader(object):
    """
    General HDF5 reader
    """
    def __init__(self, emb_file_path: str,
                 hdfs_key: str, in_memory: bool = False):
        """

        :param emb_file_path:
        :param key: hdfs key
        :param in_memory:
        """

        self.emb_file_path = emb_file_path
        self._in_memory = in_memory
        self.hdfs_key = hdfs_key

        # SA: todo `img_ids` key would change
        ## SA: trying to load everything
        if self._in_memory:
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                self.all_embeddings = embedding_hdf[self.hdfs_key][:]
                self._image_id_list = list(embedding_hdf["img_ids"])
                print(f"All embedding loaded for {self.hdfs_key}", len(self.all_embeddings))
                self.all_len = len(self.all_embeddings)
        else:
            print("Loading the file only. Not reading in memory")
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                self._split = embedding_hdf.attrs["split"]
                # SA: todo check if we can do len or shape
                self.all_len = len(embedding_hdf[self.hdfs_key])

                print(embedding_hdf.keys())
                self._image_id_list = list(embedding_hdf["img_ids"])
                # "features" is List[np.ndarray] if the dataset is loaded in-memory
                # If not loaded in memory, then list of None.
                self.all_embeddings = [None] * self.all_len

    def __len__(self):
        return len(self.all_len)


    # SA: todo check the return typing
    def __getitem__(self,
                    image_id: int):  # -> np.array:
        _index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.all_embeddings[_index] is not None:
                _embeddings = self.all_embeddings[_index]
            else:
                with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                    _embeddings = embedding_hdf[self.hdfs_key][_index]
                    # Store in memory
                    self.all_embeddings[_index] = _embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                _embeddings = embedding_hdf[self.hdfs_key][_index]

        return _embeddings

    def keys(self) -> List[int]:
        return list(range(self.all_len))

    @property
    def split(self):
        return self._split



class EmbeddingsHdfReader(object):
    """
    General HDF5 reader
    """
    def __init__(self, emb_file_path: str,
                 hdfs_key: str, in_memory: bool = False):
        """

        :param emb_file_path:
        :param key: hdfs key
        :param in_memory:
        """

        self.emb_file_path = emb_file_path
        self._in_memory = in_memory
        self.hdfs_key = hdfs_key

        ## SA: trying to load everything
        if self._in_memory:
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                self.all_embeddings = embedding_hdf[self.hdfs_key][:]
                print(f"All embedding loaded for {self.hdfs_key}", len(self.all_embeddings))
                self.all_len = len(self.all_embeddings)
        else:
            print("Loading the file only. Not reading in memory")
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                self._split = embedding_hdf.attrs["split"]
                # SA: todo check if we can do len or shape
                self.all_len = len(embedding_hdf[self.hdfs_key])
                # "features" is List[np.ndarray] if the dataset is loaded in-memory
                # If not loaded in memory, then list of None.
                self.all_embeddings = [None] * self.all_len

    def __len__(self):
        return len(self.all_len)


    # SA: todo check the return typing
    def __getitem__(self,
                    _index: int):  # -> np.array:

        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.all_embeddings[_index] is not None:
                _embeddings = self.all_embeddings[_index]
            else:
                with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                    _embeddings = embedding_hdf[self.hdfs_key][_index]
                    # Store in memory
                    self.all_embeddings[_index] = _embeddings
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.emb_file_path, "r") as embedding_hdf:
                _embeddings = embedding_hdf[self.hdfs_key][_index]

        return _embeddings

    def keys(self) -> List[int]:
        return list(range(self.all_len))

    @property
    def split(self):
        return self._split
