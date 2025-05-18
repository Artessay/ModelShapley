"""
Prompt dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_file: str, tokenizer: PreTrainedTokenizer, config: dict = {}):
        # prompt_key = config.get("prompt_key", "prompt")
        # prompt_dict_keys = config.get("prompt_dict_keys", None)
        # response_key = config.get("response_key", "response")
        # response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        prompt_key = "prompt"
        response_key = "extra_info"
        response_dict_keys = ["answer"]

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        self.parquet_file = parquet_file
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.response_dict_key = response_dict_keys[0] if response_dict_keys else "answer"

        self.max_length = max_length

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = pd.read_parquet(self.parquet_file)
        # self.dataframe = self.dataframe.head()  # debug only

        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.prompts = [prompt.tolist() for prompt in self.prompts]

        self.responses = self.dataframe[self.response_key].tolist()
        self.responses = [response[self.response_dict_key] for response in self.responses]

    @staticmethod
    def _compute_position_id_with_mask(mask):
        return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item: int):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        
        # prompt_ids_output: dict_keys(['input_ids', 'attention_mask'])
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            # Right-padding: append padding tokens after the original sequence
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = self._compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        # Right-padding: valid tokens are at the beginning of the sequence
        if prompt_length > 1:
            # Mask out prompt tokens (left side)
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # Mask out the last token in the response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }