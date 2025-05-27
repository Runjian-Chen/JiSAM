# Installation Instructions

### Clone the repository

```shell
git clone https://github.com/Runjian-Chen/JiSAM
cd JiSAM
```

### Create conda environment

```shell
conda create -n JiSAM python=3.8
conda activate JiSAM
```

### Install Pytorch

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**Note:** We tested on 1.9.0 with CUDA 11.1. But other versions should also work. Please adapt to your own hardware/system.

### Install spconv

```shell
pip install spconv-cu111
```

**Note:** Keep CUDA version of spconv consistent with Pytorch's.

### Install requirements

```shell
pip install -r requirements.txt
```

### Install pcdet

```shell
python setup.py develop
```
