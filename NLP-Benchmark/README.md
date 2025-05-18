# CFT-Benchmark

### Requirements

Create conda environment.

```bash
conda create -n verl python=3.10 -y
conda activate verl
```

And install dependencies.

```bash
pip install -r requirements.txt 
```

### Download Models

Ensure you have installed `modelscope` package.

```bash
pip install modelscope
```

Download models to local dir.

```bash
modelscope download --model LLM-Research/Llama-3.2-3B-Instruct  --local_dir /data/meta-llama/Llama-3.2-3B-Instruct
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir /data/Qwen/Qwen2.5-3B-Instruct
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /data/Qwen/Qwen2.5-7B-Instruct
modelscope download --model Qwen/Qwen2.5-14B-Instruct --local_dir /data/Qwen/Qwen2.5-14B-Instruct
```

### Running

Train:

```bash
python train.py
```

Inference:

```bash
python inference.py
```