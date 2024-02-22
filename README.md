# Simulated Event Data
Please refer to the GitHub website for v2e:https://github.com/SensorsINI/v2e?tab=readme-ov-file

# Requirements
Python 3 with the following packages installed:</br>
torch==1.9.0</br>
torchvision==0.10.0</br>
tqdm==4.62.3</br>
numpy==1.21.2</br>
imageio==2.9.0</br>
Pillow==8.3.1</br>
slayerPytorch</br>
See https://github.com/bamsumit/slayerPytorch to install the slayerPytorch for the SNN simulation.</br>

cuda</br>
A CUDA enabled GPU is required for training any model. We test our code with CUDA 11.1 V11.1.105 on NVIDIA 3090 GPUs.</br>

# Important reference
We have referred to the following articles earlier in this work. We are very grateful to the following authors for their contributions.

[1] Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), URL: https://arxiv.org/abs/2006.07722, 2021

[2] Gao, Yue, et al. "SuperFast: 200$\boldsymbol {\times} $ Video Frame Interpolation via Event Camera." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).

[3] Wang, Zhendong, et al. "Uformer: A general u-shaped transformer for image restoration." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.


# DataSet
The simulation data set needs to unzip the shapes folder and run diffvGer.m

Real time dataset obtained by contacting the author for scientific research.

# Test
First, run the TestHormer. py file and download the weight Hormer. pth.（ https://pan.baidu.com/s/13Jnbz4ZuSVHQq8G1G-vy5g?pwd=5tbz password: 5tbz） Finally, run val D. m to evaluate the ssim metric. Need to adjust the folder yourself.
