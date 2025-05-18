
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name_or_path):
    """
    load model and tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        # attn_implementation='flash_attention_2',
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
