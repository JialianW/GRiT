git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Activate conda by default
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use grit environment by default
echo "conda activate grit" >> ~/.zshrc
echo "conda activate grit" >> ~/.bashrc

# Activate conda on current shell
source /home/vscode/miniconda3/bin/activate

# Create and activate grit environment
conda create -y -n grit python=3.8
conda activate grit

echo "Installing CUDA..."
# Even though cuda package installs cuda-nvcc, it doesn't pin the same version so we explicitly set both
conda install -y -c nvidia cuda=11.3 cuda-nvcc=11.3

export CUDA_HOME=/home/vscode/miniconda3/envs/grit
echo "export CUDA_HOME=$CUDA_HOME" >> ~/.zshrc
echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc

mkdir -p models
cd models

if [ ! -f grit_b_densecap_objectdet.pth ]; then
  echo "Downloading pretrained model..."
  wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
else
  echo "Skipping download, pretrained model already exists."
fi

cd ..

echo "Installing requirements..."
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

cd third_party/CenterNet2
echo "Installing CenterNet2..."
pip install -e .
cd ../..

echo "postCreateCommand.sh completed!"
