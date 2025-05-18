"""
Prompt dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PromptDataset(Dataset):
    """
    This is an in-memory PromptDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_file: str, tokenizer: PreTrainedTokenizer, config: dict = None):
        # prompt_key = config.get("prompt_key", "prompt")
        # prompt_dict_keys = config.get("prompt_dict_keys", None)
        # response_key = config.get("response_key", "response")
        # response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        prompt_key = "prompt"
        solution_key = "reward_model"
        solution_dict_keys = ["ground_truth"]

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        self.parquet_file = parquet_file
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.solution_key = solution_key
        self.solution_dict_key = solution_dict_keys[0] if solution_dict_keys else "ground_truth"

        self.max_length = max_length

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = pd.read_parquet(self.parquet_file)
        # self.dataframe = self.dataframe.head()  # debug only

        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.prompts = [prompt.tolist() for prompt in self.prompts]

        self.solutions = self.dataframe[self.solution_key].tolist()
        self.solutions = [solution[self.solution_dict_key] for solution in self.solutions]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item: int):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return prompt_chat_str
        