import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from config import CONFIG
from dataset import PromptDataset
from utils import seed_everything, get_experiment_name

def load_model_and_tokenizer(model_path: str):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # initialize the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus)
    
    return llm, tokenizer

def inference(llm: LLM, prompts: list):
    # generate
    sampling_params = SamplingParams(n=1, max_tokens=1024, seed=42)
    outputs = llm.generate(prompts, sampling_params)

    # convert to text
    output_list = []
    for output in outputs:
        responses = [resp.text.strip() for resp in output.outputs]
        output_list.append(responses)
    return output_list


def generate(args):
    seed = args.seed
    mode = args.mode
    is_train = args.train
    model_name = args.model
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    activate_ratio = config['activate_ratio']
    activate_top_percentile = config['activate_top_percentile']
    
    experiment_name = get_experiment_name(model_name, dataset_name, mode, seed, 
                                          activate_ratio, activate_top_percentile, is_train)

    save_path = f"/data/{experiment_name}" if mode != 'vanilla' else f"/data/{model_name}"
    # if args.train:
    #     save_path = f"{save_path}/best"
        
    llm, tokenizer = load_model_and_tokenizer(save_path)

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = PromptDataset(
        parquet_file=f"data/{dataset_name}/test.parquet",
        tokenizer=tokenizer,
        config=config,
    )
    prompts = [dataset[i] for i in range(len(dataset))]

    output_list = inference(llm, prompts)

    # add to the data frame
    dataframe = dataset.dataframe
    dataframe["responses"] = output_list

    # write to a new parquet
    output_path = f"outputs/{experiment_name}.parquet"
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    dataframe.to_parquet(output_path)


if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    generate(args)