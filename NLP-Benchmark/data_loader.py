
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, random_split

from dataset import PromptDataset, SFTDataset

def get_data_loader(dataset_name: str, tokenizer: PreTrainedTokenizer, config: dict):
    
    full_dataset = SFTDataset(
        parquet_file=f"data/{dataset_name}/train.parquet",
        tokenizer=tokenizer,
        config=config,
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    test_dataset = PromptDataset(
        parquet_file=f"data/{dataset_name}/test.parquet",
        tokenizer=tokenizer,
        config=config,
    )

    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader, test_loader