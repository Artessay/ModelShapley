CONFIG = {
    'gsm8k': {
        # 'epochs': 1,
        # 'lr': 3e-5,
        # 'weight_decay': 0.0,
        'batch_size': 8,
        'max_length': 1024,
        "truncation": "error",
        'activate_ratio': 0.1,
        'activate_top_percentile': True,
    },
    'mmlu': {
        # 'epochs': 1,
        # 'lr': 3e-5,
        # 'weight_decay': 0.0,
        'batch_size': 8,
        'max_length': 2048,
        "truncation": "error",
        'activate_ratio': 0.1,
        'activate_top_percentile': True,
    },
}