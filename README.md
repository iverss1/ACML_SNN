# ECML_SNN
Training a General Spiking Neural Network with Improved Efficiency and Minimum Latency
this is the accompanied code of submission of *Training a General Spiking Neural Network with Improved Efficiency and Minimum Latency*:
## Setup
If you are using Anaconda, you might want to use virtual environment.
## Train a SNN model

## Validate a SNN model
To prove our research, we provide the pre-train file of two versions (CNN-SNN/LSTM-SNN) on the dataset CIFAR10 and CIFAR100.
Please download the corresponding dataset for validation.
### 1(a). If you wanna test CNN-SNN model
#### 1.Import CNN-SNN layer in ```two_direction_snn.py```
```bash
from models.layers_ecml_cnn_10 import Sequencer2DBlock, PatchEmbed, Downsample2D,SNN2D,PatchMerging
```
#### 2.Modify related argument in ```load_model.py```
```bash
'--dataset', default='torch/cifar10',help='dataset type cifar10 or cifar100'
'--load-model', default='/home/yj/yyp/SNN_CV/output/cifar_10_not_best_model_best.pth.tar',help='dataset type cifar10 or cifar100'
```
### 1(b). If you wanna test LSTM-SNN model
### 2. Load the model to view relevant results (Acc, Macs, Structure and etc)
```bash
Python load_model.py
```
