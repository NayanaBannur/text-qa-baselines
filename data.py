import json
import logging
import os
import re
import sys
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def read_webqa_json(path: str=None, split: str='train'):
    if path is None:
        raise ValueError("Must specify data file path.")
    with open(path, "r") as f:
        data = json.load(f)

        ids, questions, answers, qu_category, topic, img_pos_captions, img_neg_captions, txt_pos, txt_neg \
            = [], [], [], [], [], [], [], [], []
        
        count = 0
        for key, info in data.items():
            if info['split'] == split:
                count += 1
                ids.append(key)
                questions.append(info['Q'].replace('"', ""))
                answers.append(info['A'][0].replace('"', ""))
                if split in ['train', 'val']:
                    qu_category.append(info['Qcate'])
                    topic.append(info['topic'])
                    img_pos_captions_curr = []
                    for img_data in info['img_posFacts']:
                        img_pos_captions_curr.append(img_data['caption'])
                    img_pos_captions.append(img_pos_captions_curr)
                    img_neg_captions_curr = []
                    for img_data in info['img_negFacts']:
                        img_neg_captions_curr.append(img_data['caption'])
                    img_neg_captions.append(img_neg_captions_curr)
                    txt_pos_curr = []
                    for txt_data in info['txt_posFacts']:
                        txt_pos_curr.append(txt_data['fact'])
                    txt_pos.append(txt_pos_curr)
                    txt_neg_curr = []
                    for txt_data in info['txt_posFacts']:
                        txt_neg_curr.append(txt_data['fact'])
                    txt_neg.append(txt_neg_curr)
                else:
                    img_pos_captions_curr = []
                    for img_data in info['img_Facts']:
                        img_pos_captions_curr.append(img_data['caption'])
                    img_pos_captions.append(img_pos_captions_curr)
                    txt_pos_curr = []
                    for txt_data in info['txt_Facts']:
                        txt_pos_curr.append(txt_data['fact'])
                    txt_pos.append(txt_pos_curr)
            if count>5:
                break

    return ids, questions, answers, qu_category, topic, img_pos_captions, img_neg_captions, txt_pos, txt_neg


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]


def convert_text(text):
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text.lower()


def get_lines(fil):
    lines = []
    with open(fil, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
            else:
                lines.append('empty')
    return lines


def create_input(tokenizer, max_length,
                 questions, answers, qu_category, topic,
                 img_pos_captions, img_neg_captions, txt_pos, txt_neg,
                 input_type='qa', padding=True, return_tensors="pt"):
    examples = []
    if input_type == 'qa':
        for i, _ in enumerate(tqdm(questions)):
            text = questions[i]
            tokenized = tokenizer.batch_encode_plus(
                [text.strip()], truncation=True, max_length=max_length, padding=padding, return_tensors=return_tensors)
            examples.append(tokenized)
    return examples


def create_output(tokenizer, max_length,
                  questions, answers, qu_category, topic,
                  img_pos_captions, img_neg_captions, txt_pos, txt_neg,
                  input_type='qa', padding=True, return_tensors="pt"):
    examples = []
    if input_type == 'qa':
        for i, _ in enumerate(tqdm(questions)):
            text = answers[i]
            tokenized = tokenizer.batch_encode_plus(
                [text.strip()], truncation=True, max_length=max_length, padding=padding, return_tensors=return_tensors)
            examples.append(tokenized)
    return examples


class WebQATextDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./", filepath=None, split="train",
                 max_source_length=768, max_target_length=512):
        super().__init__()
        if filepath is None:
            raise ValueError("Must specify data file path.")
        ids, questions, answers, qu_category, topic, img_pos_captions, img_neg_captions, txt_pos, txt_neg \
            = read_webqa_json(os.path.join(data_dir, filepath), split=split)
        
        self.tokenizer = tokenizer
        self.source = create_input(tokenizer, max_source_length, questions, answers, qu_category, topic,
                                   img_pos_captions, img_neg_captions, txt_pos, txt_neg)
        self.target = create_output(tokenizer, max_target_length, questions, answers, qu_category, topic,
                                   img_pos_captions, img_neg_captions, txt_pos, txt_neg)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}


if __name__ == "__main__":
    pass

