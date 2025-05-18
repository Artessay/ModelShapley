# Copyright 2025 Rihong Qiu
"""
Preprocess the MMLU dataset to parquet format
"""

import argparse
import os

import datasets


def format_question(item):
    question = item["question"]
    options = item["choices"]

    prompt = "Question:\n"
    prompt += question + "\n"
    prompt += "Options:\n"
    choice_list = []
    for i, opt in enumerate(options):
        choice = chr(ord('A') + i)
        choice_list.append("{}. {}".format(choice, opt))
    prompt += "\n".join(choice_list)

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./mmlu")

    args = parser.parse_args()

    data_source = "cais/mmlu"

    dataset = datasets.load_dataset(data_source, "all")

    train_dataset = dataset["auxiliary_train"]
    test_dataset = dataset["test"]

    instruction_following = """The following is a multiple choice question. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice."""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = format_question(example)

            solution = chr(ord('A') + example["answer"])
            answer = f"The answer is ({solution})"
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": instruction_following,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "knowledge",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
