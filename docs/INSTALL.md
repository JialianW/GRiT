# Installation

Our experiments are based on Python 3.8, PyTorch 1.9.0, and CUDA 11.1. 
Newer version of PyTorch may also work.

### Installation Outline

- Download and install Pytorch
- Download and install Detectron2. We use the version at the commit ID cc87e7ec
- Download and install GRiT

### Installation Example
```bash
conda create --name grit python=3.8 -y
conda activate grit
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout cc87e7ec
pip install -e .

cd ..
git clone https://github.com/JialianW/GRiT.git
cd GRiT
pip install -r requirements.txt
```