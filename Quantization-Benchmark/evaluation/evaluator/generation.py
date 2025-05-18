import os
import hydra
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def inference(model_path: str, chat_list: list):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # initialize the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus)

    # convert to chat template format
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chat_list]

    # generate
    sampling_params = SamplingParams(n=1, max_tokens=1024, seed=42)
    outputs = llm.generate(prompts, sampling_params)

    # convert to text
    output_list = []
    for output in outputs:
        responses = [resp.text.strip() for resp in output.outputs]
        output_list.append(responses)
    return output_list


@hydra.main(config_path="config", config_name="generation", version_base=None)
def generate(config):
    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_list = dataset[config.data.prompt_key].tolist()

    chat_list = [chat.tolist() for chat in chat_list]
    output_list = inference(config.model.path, chat_list)

    # add to the data frame
    dataset[config.data.response_key] = output_list

    # write to a new parquet
    output_path = config.data.output_path
    output_dir = os.path.dirname(config.data.output_path)
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    generate()