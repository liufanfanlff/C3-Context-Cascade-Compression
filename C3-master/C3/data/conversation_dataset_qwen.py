
import io
import os
import copy
import json
import logging
import torch
import random

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from C3.data.base_dataset import BaseDataset
from C3.utils.constants import *
from C3.utils import conversation as conversation_lib
import boto3
import smart_open
from megfile import smart_glob
from natsort import natsorted


class ConversationDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, data_path, tokenizer, multimodal_cfg):
        super(ConversationDataset, self).__init__(data_path, tokenizer, multimodal_cfg)
        # v0 version format conversation
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        logging.warning("Formatting inputs into conversation type: mpt-fixed")
        logging.warning("Loading data...")
        list_data_dict = []
        data = json.load(open(data_path, "r"))
        list_data_dict.extend(data)
        logging.warning(f"Data from {data_path} provide {len(data)} conversations.")
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        random.shuffle(list_data_dict)
        list_data_dict_new = list_data_dict
        self.list_data_dict = list_data_dict_new
        self.im_patch_token = 151859
        self.im_start_token = 151857
        self.im_end_token = 151858
    
    def processor(self, sources, flag_num_patches):
        sources_processor = []
        for source in sources:
            source_processor = []
            for sentence in source:
                if sentence['from'] == 'context':
                    continue
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']*flag_num_patches
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
                source_processor.append(sentence)
            sources_processor.append(source_processor)
        return sources_processor

    def processor_context(self, sources, flag_num_patches):
        sources_processor = []
        for source in sources:
            source_processor = []
            for sentence in source:
                if sentence['from'] == 'context':
                        
                    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']*flag_num_patches
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    sentence["value"] = str(sentence["value"]) + replace_token
                    source_processor.append(sentence)
            sources_processor.append(source_processor)
        return sources_processor

    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def _mask_targets(self, target, tokenized_lens, speakers):
        # cur_idx = 0
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker.lower() == "human":
                target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

    def token_processor(self, sources, sources_context):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        conversations_context = []

        for i, source in enumerate(sources_context):
            conversations_context.append(sources_context[0][0]['value'])


        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length = self.tokenizer.model_max_length+1000,
            truncation=True,
        ).input_ids

        assert input_ids.shape[1] < self.tokenizer.model_max_length

        context_ids = self.tokenizer(
            conversations_context,
            return_tensors="pt",
            padding="longest",
            max_length = self.tokenizer.model_max_length + 1000,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids) + len(self.tokenizer(conv.sep).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids)
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                  

        return dict(
            input_ids=input_ids,
            labels=targets,
            context_ids=context_ids,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = copy.deepcopy(self.list_data_dict[i])

        if isinstance(data, dict):
            image_list =  []
            image_high_list = []
            flag_num_patches = 1
            conversations_context = self.processor_context([copy.deepcopy(data["conversations"])], flag_num_patches)
            conversations = self.processor([copy.deepcopy(data["conversations"])], flag_num_patches)
        else:
            conversations = [data]

        try:
            data_dict = self.token_processor(conversations, conversations_context)
        except:
            print(f'len out!')
            return self.__getitem__(random.randint(0, self.__len__()-1))

        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0],context_ids = data_dict["context_ids"][0])

        return data_dict

