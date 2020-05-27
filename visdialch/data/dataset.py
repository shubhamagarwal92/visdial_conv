from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from visdialch.data.readers import (
    DialogsReader,
    DenseAnnotationsReader,
    ImageFeaturesHdfReader,
    TransformerEmbeddingsHdfReader,
    QuesEmbeddingsHdfReader,
    AnswerEmbeddingsHdfReader,
    CaptionEmbeddingsHdfReader,
    HistEmbeddingsHdfReader,
    EmbeddingsHdfReader,
    AugmentedDenseAnnotationsReader
)
from visdialch.data.vocabulary import Vocabulary
import time
import numpy as np

# SA: try this for multiprocessing
# https://github.com/pytorch/pytorch/issues/973
# SA: todo try and remove this
# torch.multiprocessing.set_sharing_strategy('file_system')

class VisDialDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, ground truth answer, answer options, dense annotations etc.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dialogs_jsonpath: str,
        dense_annotations_jsonpath: Optional[str] = None,
        augment_dense_annotations_jsonpath: Optional[str] = None,
        use_pretrained_emb: bool = False,
        qa_emb_file_path: Optional[str] = None,  # SA: todo remove this
        hist_emb_file_path: Optional[str] = None,  # SA: todo remove this
        use_caption: bool = True,
        num_hist_turns: int = 10,
        finetune: bool = False,
        overfit: bool = False,
        in_memory: bool = False,
        num_workers: int = 1,
        return_options: bool = True,
        add_boundary_toks: bool = False):

        super().__init__()
        self.config = config

        # SA: embedding reader
        self.use_pretrained_emb = use_pretrained_emb

        self.return_options = return_options
        self.add_boundary_toks = add_boundary_toks
        self.dialogs_reader = DialogsReader(
            dialogs_jsonpath,
            num_examples=(5 if overfit else None),
            num_workers=num_workers,
            use_pretrained_emb=self.use_pretrained_emb
        )

        self.finetune = finetune
        self.use_caption = use_caption

        # SA: embedding reader
        if self.use_pretrained_emb:
            assert qa_emb_file_path , "Did you forget to set emb file path?"
            # @todo: for now coming through argparse
            self.qa_emb_file_path = qa_emb_file_path
            self.hist_emb_file_path = hist_emb_file_path
            # hist_emb_file_path = config["hist_emb_file_path"]
            # TransformerEmbeddingsHdfReader(embedding_path, in_memory)
            # self.embedding_reader = TransformerEmbeddingsHdfReader(hist_emb_file_path,
            #                                                        in_memory)
            self.question_reader = QuesEmbeddingsHdfReader(qa_emb_file_path, in_memory)
            self.ans_reader = AnswerEmbeddingsHdfReader(qa_emb_file_path, in_memory)
            self.caption_reader = CaptionEmbeddingsHdfReader(qa_emb_file_path, in_memory)

            # SA: we dont pass in_memory here because history is too big
            # SA: todo this key would change
            self.hist_reader = HistEmbeddingsHdfReader(hist_emb_file_path, hdfs_key="hist")


        # SA: if finetuning for train/val  otherwise just validation set
        if self.finetune or ("val" in self.split and dense_annotations_jsonpath is not None):
            self.annotations_reader = DenseAnnotationsReader(
                dense_annotations_jsonpath
            )
        else:
            self.annotations_reader = None

        if augment_dense_annotations_jsonpath is not None:
            self.augmented_annotations_reader = AugmentedDenseAnnotationsReader(
                augment_dense_annotations_jsonpath
            )
            self.use_augment_dense = True
        else:
            self.use_augment_dense = False

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # Initialize image features reader according to split.
        image_features_hdfpath = config["image_features_train_h5"]
        if "val" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_val_h5"]
        elif "test" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_test_h5"]

        self.hdf_reader = ImageFeaturesHdfReader(
            image_features_hdfpath, in_memory
        )

        # Keep a list of image_ids as primary keys to access data.
        # For finetune we use only those image id where we have dense annotations
        if self.finetune:
            self.image_ids = list(self.annotations_reader.keys)
        else:
            self.image_ids = list(self.dialogs_reader.dialogs.keys())

        if overfit:
            self.image_ids = self.image_ids[:5]


    @property
    def split(self):
        return self.dialogs_reader.split

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # start = time.time()
        # Get image_id, which serves as a primary key for current instance.
        image_id = self.image_ids[index]

        # Get image features for this image_id using hdf reader.
        image_features = self.hdf_reader[image_id]
        image_features = torch.tensor(image_features)
        # Normalize image features at zero-th dimension (since there's no batch
        # dimension).
        if self.config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)

        # Retrieve instance for this image_id using json reader.
        visdial_instance = self.dialogs_reader[image_id]
        caption = visdial_instance["caption"]
        dialog = visdial_instance["dialog"]

        # SA: reading embeddings here
        if self.use_pretrained_emb:
            # We need indexes to actually call the readers here now.
            dialog_with_index = visdial_instance["dialog_with_index"]
            original_index = visdial_instance["original_index"]
            assert len(dialog) == len(dialog_with_index), "These should be equal => just saving the index instead of string"

        # ideally should be in if-else clause
        ques_embeddings = []
        ans_embeddings = []
        opts_embeddings = []

        # Convert word tokens of caption, question, answer and answer options
        # to integers.
        caption = self.vocabulary.to_indices(caption)
        for i in range(len(dialog)):

            # SA: using embeddings here in the same loop
            if self.use_pretrained_emb:

                # SA: todo We dont need caption embeddings when we already have history???
                # caption_embedding = self.caption_reader[original_index]
                ques_embeddings.append(self.question_reader[dialog_with_index[i]["question"]])
                ans_embeddings.append(self.ans_reader[dialog_with_index[i]["answer"]])

            # SA: original code
            dialog[i]["question"] = self.vocabulary.to_indices(
                dialog[i]["question"]
            )
            if self.add_boundary_toks:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    [self.vocabulary.SOS_TOKEN]
                    + dialog[i]["answer"]
                    + [self.vocabulary.EOS_TOKEN]
                )
            else:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    dialog[i]["answer"]
                )

            # for disc decoder
            if self.return_options:
                # Ideally should be in if-else clause
                opts_round_embeddings = []

                for j in range(len(dialog[i]["answer_options"])):

                    # SA: trying option encodings here now
                    if self.use_pretrained_emb:
                        opts_round_embeddings.append(self.ans_reader[dialog_with_index[i]["answer_options"][j]])

                    if self.add_boundary_toks:
                        dialog[i]["answer_options"][
                            j
                        ] = self.vocabulary.to_indices(
                            [self.vocabulary.SOS_TOKEN]
                            + dialog[i]["answer_options"][j]
                            + [self.vocabulary.EOS_TOKEN]
                        )
                    else:
                        dialog[i]["answer_options"][
                            j
                        ] = self.vocabulary.to_indices(
                            dialog[i]["answer_options"][j]
                        )

                # Ideally should be in if-else clause
                opts_embeddings.append(opts_round_embeddings)

        questions, question_lengths = self._pad_sequences(
            [dialog_round["question"] for dialog_round in dialog]
        )
        history, history_lengths = self._get_history(
            caption,
            [dialog_round["question"] for dialog_round in dialog],
            [dialog_round["answer"] for dialog_round in dialog],
        )
        answers_in, answer_lengths = self._pad_sequences(
            [dialog_round["answer"][:-1] for dialog_round in dialog]
        )
        answers_out, _ = self._pad_sequences(
            [dialog_round["answer"][1:] for dialog_round in dialog]
        )

        # Collect everything as tensors for ``collate_fn`` of dataloader to
        # work seamlessly questions, history, etc. are converted to
        # LongTensors, for nn.Embedding input.
        item = {}
        item["img_ids"] = torch.tensor(image_id).long()
        item["img_feat"] = image_features
        item["ques"] = questions.long()
        item["hist"] = history.long()
        item["ans_in"] = answers_in.long()  # SA: probably useful for training gen
        item["ans_out"] = answers_out.long()  # SA: probably useful for training gen
        item["ques_len"] = torch.tensor(question_lengths).long()
        item["hist_len"] = torch.tensor(history_lengths).long()
        item["ans_len"] = torch.tensor(answer_lengths).long()
        item["num_rounds"] = torch.tensor(visdial_instance["num_rounds"]).long()

        ## SA: pretrained embedding here
        if self.use_pretrained_emb:

            # See https://github.com/pytorch/pytorch/issues/13918
            item["ques_embeddings"] = torch.tensor(np.array(ques_embeddings)).float()
            # now (10, 20, 768) ==> will be (bs, 10, 20, 768) (bert embeddings)
            item["opts_embeddings"] = torch.tensor(np.array(opts_embeddings)).float()
            # ans_embeddings = torch.tensor(np.array(ans_embeddings)).float()
            # caption_embedding = torch.tensor(np.array(caption_embedding)).float()

            # SA: todo proxy hist embeddings
            # hist_embeddings = self._get_history_embedding(caption_embedding, item["ques_embeddings"],
            #                                               ans_embeddings)
            item["hist_embeddings"] = self.hist_reader[image_id]
            # (10, 100, 20, 768) ==> will be (bs, 10, 100, 20, 768) (bert embeddings)

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        [
                            option[:-1]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_in.append(options)

                    options, _ = self._pad_sequences(
                        [
                            option[1:]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_out.append(options)

                    answer_option_lengths.append(option_lengths)
                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)

                item["opt_in"] = answer_options_in.long()
                item["opt_out"] = answer_options_out.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()
            else:
                answer_options = []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        dialog_round["answer_options"]
                    )
                    answer_options.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options = torch.stack(answer_options, 0)

                # used by disc model
                ## options_length SA: used by model to select non-zero options
                item["opt"] = answer_options.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()

            if "test" not in self.split:
                answer_indices = [
                    dialog_round["gt_index"] for dialog_round in dialog
                ]
                item["ans_ind"] = torch.tensor(answer_indices).long()  # Used by evaluate for ndcg

        # Gather dense annotations.
        if self.finetune or ("val" in self.split):
            dense_annotations = self.annotations_reader[image_id]

            # SA: have to do this because of changed dic key in train
            if "val" in self.split:
                item["gt_relevance"] = torch.tensor(
                    dense_annotations["gt_relevance"]
                ).float()
            elif "train" in self.split:
                item["gt_relevance"] = torch.tensor(
                    dense_annotations["relevance"]
                ).float()

            item["round_id"] = torch.tensor(
                dense_annotations["round_id"]
            ).long()
        # end = time.time()
        # time_taken = end - start
        # print('Time for loading item: ',time_taken)

        if self.use_augment_dense:
            augmented_dense_annotations = self.augmented_annotations_reader[image_id]

            item["augmented_gt_relevance"] = torch.tensor(
                augmented_dense_annotations["augmented_gt_relevance"]
            ).float()



        return item

    def _pad_sequences(self, sequences: List[List[int]]):
        """Given tokenized sequences (either questions, answers or answer
        options, tokenized in ``__getitem__``), padding them to maximum
        specified sequence length. Return as a tensor of size
        ``(*, max_sequence_length)``.

        This method is only called in ``__getitem__``, chunked out separately
        for readability.

        Parameters
        ----------
        sequences : List[List[int]]
            List of tokenized sequences, each sequence is typically a
            List[int].

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences
            before padding.
        """

        for i in range(len(sequences)):
            sequences[i] = sequences[i][
                : self.config["max_sequence_length"] - 1
            ]
        sequence_lengths = [len(sequence) for sequence in sequences]

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full(
            (len(sequences), self.config["max_sequence_length"]),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths

    def _get_history(
        self,
        caption: List[int],
        questions: List[List[int]],
        answers: List[List[int]],
    ):
        # Allow double length of caption, equivalent to a concatenated QA pair.
        caption = caption[: self.config["max_sequence_length"] * 2 - 1]

        for i in range(len(questions)):
            questions[i] = questions[i][
                : self.config["max_sequence_length"] - 1
            ]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.config["max_sequence_length"] - 1]

        # History for first round is caption, else concatenated QA pair of
        # previous round.
        history = []
        ## SA: appending EOS after caption
        caption = caption + [self.vocabulary.EOS_INDEX]
        if self.use_caption:
            history.append(caption)
        else:
            history.append([self.vocabulary.EOS_INDEX])
            # print("Not using caption in history.")
        for question, answer in zip(questions, answers):
            history.append(question + answer + [self.vocabulary.EOS_INDEX])
        # Drop last entry from history (there's no eleventh question).
        history = history[:-1]
        max_history_length = self.config["max_sequence_length"] * 2

        if self.config.get("concat_history", False):
            # Concatenated_history has similar structure as history, except it
            # contains concatenated QA pairs from previous rounds.
            concatenated_history = []
            concatenated_history.append(caption)
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i + 1):
                    concatenated_history[i].extend(history[j])

            max_history_length = (
                self.config["max_sequence_length"] * 2 * len(history)
            )
            history = concatenated_history

        history_lengths = [len(round_history) for round_history in history]
        maxpadded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_history[:, : padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths


    def _get_history_embedding(self, caption,
                               questions,
                               answers):
        """
        only for one dialogue here
        num_rounds = 10

        :param caption: (40, 768) ==> cross check
        :param questions: (10, 20, 768)
        :param answers: (10, 20, 768)
        :return:
        """

        concatenated_qa_history = torch.cat([questions, answers], 1)
        # print(concatenated_qa_history.size())
        # Drop last
        concatenated_qa_history = concatenated_qa_history[:-1]
        caption = caption.unsqueeze(0)
        # Concatenate along batch now
        concatenated_qa_history = torch.cat([caption, concatenated_qa_history], 0)  # shape (10, 40, 768)


        if self.config.get("concat_history", False):
            max_history_length = (self.config["max_sequence_length"] * 2 * len(concatenated_qa_history))  # 400

            history_list = []
            num_rounds , _, rep_size = concatenated_qa_history.size()  # (10, 40, 768)

            # hist_tensor = concatenated_qa_history.view(-1, rep_size)  # (10*40, 768)
            # hist_tensor = hist_tensor.unsqueeze(0).repeat(num_rounds,1,1)  # (10, 400, 768)
            # zero_array =


            for i in range(1, num_rounds+1):

                pad_array = torch.zeros(max_history_length - self.config["max_sequence_length"] * 2 * (i), rep_size)

                hist_array = concatenated_qa_history[:i].view(-1, rep_size)

                hist_round = torch.cat([hist_array, pad_array], 0)
                history_list.append(hist_round)

            history = torch.stack(history_list,0)
        else:
            history = concatenated_qa_history


        return history

    def _get_combined_ques_caption_or_hist(self,
                                           caption: List[int],
                                           questions: List[List[int]],
                                           answers: List[List[int]]
                                           ):
        # Allow double length of caption, equivalent to a concatenated QA pair.
        caption = caption[: self.config["max_sequence_length"] * 2 - 1]

        for i in range(len(questions)):
            questions[i] = questions[i][
                : self.config["max_sequence_length"] - 1
            ]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.config["max_sequence_length"] - 1]
