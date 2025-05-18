# ArtQuantization

This repository contains the implementation of ArtQuantization, a novel approach for model quantization that combines GPTQ with Shapley value correction.

## Features

- Multiple quantization methods:
  - GPTQ-based quantization
  - OBS (Optimal Brain Surgeon) quantization
  - Shapley value-based quantization
- Support for various model architectures
- Configurable quantization parameters
- Logging and result tracking

## Requirements

- Python 3.8+
- PyTorch
- llmcompressor
- Other dependencies (see requirements.txt)

## Usage

### GPTQ Quantization

To run GPTQ-based quantization:

```bash
python quantize_by_gptq.py \
    --model_path [path_to_model] \
    --scheme [quantization_scheme] \
    --dataset [dataset_name] \
    --dataset_config [config_name] \
    --num_calibration_samples [num_samples] \
    --alpha [alpha_value] \
    --seed [random_seed]
```

Or use the provided script:
```bash
bash script/run_gptq.sh
```

### OBS Quantization

To run OBS-based quantization:

```bash
python quantize_by_obs.py \
    --model_path [path_to_model] \
    --scheme [quantization_scheme] \
    --dataset [dataset_name] \
    --dataset_config [config_name] \
    --num_calibration_samples [num_samples] \
    --seed [random_seed]
```

Or use the provided script:
```bash
bash script/run_obs.sh
```

### Shapley Value Quantization

To run Shapley value-based quantization:

```bash
python quantize_by_shapley.py \
    --model_path [path_to_model] \
    --scheme [quantization_scheme] \
    --dataset [dataset_name] \
    --dataset_config [config_name] \
    --num_calibration_samples [num_samples] \
    --seed [random_seed]
```

Or use the provided script:
```bash
bash script/run_shapley.sh
```

## Parameters

- `model_path`: Path to the model to be quantized
- `scheme`: Quantization scheme to use
- `dataset`: Name of the dataset for calibration
- `dataset_config`: Configuration name for the dataset (default: "main")
- `num_calibration_samples`: Number of samples to use for calibration (default: 512)
- `alpha`: Alpha parameter for GPTQ (default: 0.01)
- `seed`: Random seed for reproducibility (default: 42)

## Output

The quantized model will be saved in a directory named according to the following pattern:
```
{model_path}-{method}-{scheme}-{dataset}-num-samples-{num_samples}-alpha-{alpha}-v1-fl32
```

A log file `result-time.log` will be created to track the execution time and parameters of each run.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
