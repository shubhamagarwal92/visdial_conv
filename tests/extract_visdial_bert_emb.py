import argparse
from tqdm import tqdm
from typing import Any, Dict
import json
import pickle
import numpy as np
import torch
import sys
sys.path.append("../../")
from visdialch.data.get_bert_embeddings import BertEmbedder

class EmbeddingExtractor:
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.save_data_dir = config.save_data_dir
        self.device = config.device
        self.embedder = BertEmbedder(device=self.device)

    def extract_emb(self, type: str):
        data_path = self._get_pkl_path(type)
        # SA: working example of data loading
        # Test load
        print("Trying to load raw text now")
        with open(data_path, 'rb') as fb:
            pickle_obj = pickle.load(fb)

        questions = pickle_obj["questions"]
        answers = pickle_obj["answers"]
        captions = pickle_obj["captions"]
        print("Number of questions: {}".format(len(questions)))
        print("Number of answers: {}".format(len(answers)))
        print("Number of captions: {}".format(len(captions)))

        question_embedding = self.embedder.get_bert_embeddings(questions, max_seq_len=20)
        print("Question embedding size: {}".format(question_embedding.size()))
        answer_embedding = self.embedder.get_bert_embeddings(answers, max_seq_len=20)
        print("Answer embedding size: {}".format(answer_embedding.size()))

        caption_embedding = self.embedder.get_bert_embeddings(captions, max_seq_len=40)
        print("Caption embedding size: {}".format(caption_embedding.size()))

        embedding_json ={
            "question_embedding": question_embedding,
            "amnswer_embedding": answer_embedding,
            "caption_embedding": caption_embedding
        }

        save_file_path = self._save_file_path(type)
        with open(save_file_path, 'wb') as outfile:
            pickle.dump(embedding_json, outfile)

        # save_file_path = self._save_file_path(type=type, ext=".pth")
        # torch.save(embedding_json, save_file_path)


    def _get_pkl_path(self, type: str,
                       split: str = '1.0') -> str:
        json_path = "{}/visdial_{}_{}_raw_text.pkl".format(self.data_dir, split, type)
        return json_path

    def _save_file_path(self, type: str,
                        split: str = '1.0',
                        ext: str = 'pkl') -> str:
        file_path = "{}/visdial_{}_{}_bert_emb.{}".format(self.save_data_dir, split, type, ext)
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

    parser.add_argument(
        "-g", "--gpu-ids",
        nargs="+",
        type=int,
        default=0,
        help="List of ids of GPUs to use.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    if isinstance(config.gpu_ids, int):
        config.gpu_ids = [config.gpu_ids]
    config.device = (
        torch.device("cuda", config.gpu_ids[0])
        if config.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    extractor = EmbeddingExtractor(config)
    extractor.extract_emb("val")
    extractor.extract_emb("test")
    extractor.extract_emb("train")
