# ArtTuner

### Requirements

Create conda environment:

```bash
conda create -n artuner python==3.11 -y
conda activate artuner
```

Install PyTorch and Flash Attention libraries:

```bash
# install verl together with some lightweight dependencies in setup.py
pip3 install torch==2.6.0
# For flash attention 2, you can also download it from https://github.com/Dao-AILab/flash-attention/releases, and install it manually. Otherwise, you can install it via `pip3 install flash-attn --no-build-isolation`
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```