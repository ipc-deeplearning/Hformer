from torchvision.utils import save_image
from torch import optim, floor
from torch.utils.data import DataLoader
from OptimUtil import *
from src.loss.ssimLoss import SSIM
import cv2
import torch
import numpy as np
import torch.nn as nn

from src.model.Uformer import Uformer
from src.model.UformerY import UformerY
import argparse, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import os
import random
import torch
from src.datahandler.denoise_dataset2 import DenoiseDataSet
len = 500
path = '/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time7/HoloN/Img'

class flower256(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):

        dataset_path = os.path.join(path)
        # dataset_path = os.path.join('/data/tmj/LGBnet/dataset/old_crop')

        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

        self.img_paths.sort()

    def _load_data(self, data_idx):

        file_name1 = self.img_paths[data_idx]
        noisy_img1 = self._load_img(file_name1)

        file_name2 = self.img_paths[data_idx+1]
        noisy_img2 = self._load_img(file_name2)

        file_name3 = self.img_paths[data_idx+2]
        noisy_img3 = self._load_img(file_name3)


        rd1 = random.randint(0, len-5)  #取前面的


        file_name4 = self.img_paths[rd1]
        noisy_img4 = self._load_img(file_name4)


        noisyImage = torch.cat((noisy_img1,noisy_img2,noisy_img3),0)
        noisyImage2 = torch.cat((noisy_img4,noisy_img4,noisy_img4),0)

        # 进行随机裁剪为256大小
        # 进行随机裁剪为256大小
        Hr = random.randint(0, 255)
        Wr = random.randint(0, 255)

        noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]
        noisyImage2 = noisyImage2[:,Hr:Hr+256, Wr:Wr+256]


        return {'real_noisy1': noisyImage,'real_noisy2': noisyImage2} # only noisy image dataset

class flowershow(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):


        dataset_path = os.path.join('/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time7/HoloN/Img')

        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

        self.img_paths.sort()


    def _load_data(self, data_idx):

        file_name1 = self.img_paths[data_idx]
        noisy_img1 = self._load_img(file_name1)

        file_name2 = self.img_paths[data_idx+1]
        noisy_img2 = self._load_img(file_name2)

        file_name3 = self.img_paths[data_idx+2]
        noisy_img3 = self._load_img(file_name3)

        noisyImage = torch.cat((noisy_img1,noisy_img2,noisy_img3),0)

        # 进行随机裁剪为256大小
        # 进行随机裁剪为256大小
        Hr = 30
        Wr = 30

        noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]

        # clean_img = self._load_img(file_name)

        # return {'clean': clean_img, 'real_noisy': noisy_img} # paired dataset
        return {'real_noisy1': noisyImage} # only noisy image dataset


# saved_checkpoint = torch.load('/data/tmj/FakerFlower/RES/UandYformerCompare/UformerNoise/model/14.pth')
# saved_checkpoint = torch.load('/data/tmj/FakerFlower/RES/NewShipingFlower/model/35.pth')
# saved_checkpoint = torch.load('/data/tmj/FakerFlower/test2/noPre2/model99.pth')

depths = [4, 4, 4, 4, 4, 4, 4, 4, 4]
module = Uformer(img_size=256, embed_dim=16, depths=depths,
                            win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
                            shift_flag=False)
module = module.cuda()


module.load_state_dict(torch.load("/data/40/ntmj/ntmj/eventMain/out/UformerRessmall/model/98.pth"),strict=False)
# moduleB.load_state_dict(saved_checkpointB,strict=True)

# LossS = nn.L1Loss(reduction='mean').cuda()
l1loss = nn.L1Loss(reduction='mean').cuda()

# l1loss = nn.CrossEntropyLoss()
# op = optim.AdamW(module.parameters(),lr = 1e-4,weight_decay=1e-5)#定义优化器
train_data = flower256()
bs = 3
train_loader = DataLoader(train_data,batch_size=1,shuffle=False,drop_last=True)


ssim_loss = SSIM().cuda()

start = 50
end = 100
Savepath = '/data/40/ntmj/ntmj/eventMain/out/UformerRes/'


op1 = optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)  # 定义优化器
import time

for epoch in range(start,end):
    module.train()


    for batch_id, x in enumerate(train_loader):

        noiseImg = x['real_noisy1'].cuda()
        label = x['real_noisy2'].cuda()

        preS = label[:,1,:,:].unsqueeze(1)

        start_time = time.time()

        nowS, nowD = module(noiseImg)

        end_time = time.time()
        print(end_time-start_time)

        cc = torch.reshape(nowS,(1,-1))
        max = torch.mode(cc)[0]
        pt = noiseImg[:,1,:,:].unsqueeze(1)
        loss1 = l1loss(nowS, preS)

        ssim_out2 = 1 - ssim_loss(pt,nowD*(nowS/max))
        losstoal = loss1+ ssim_out2
        op1.zero_grad()
        losstoal.backward()
        op1.step()

        # print("epoch:   ",epoch, "loss1:",loss1.item()," loss3",ssim_out2.data.item())
        # print("epoch",epoch,"   loss2：",ssim_out2.data.item())



    module.eval()
    testdata = flowershow()
    if epoch >=20:

        for i in range(70):

            x = testdata.__getitem__(i)['real_noisy1'].cuda()
            noise = x[1]
            folder_path = Savepath+'/picture/' + str(epoch)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)

            # save_image(noise, folder_path + '/' + str(i) + '_N.png')
            x = x.unsqueeze(0)
            S, D = module(x)

            cc = torch.reshape(S, (1, -1))
            max = torch.mode(cc)[0]
            pt = x[:, 1, :, :].unsqueeze(1)

            save_image(S, folder_path + '/' + str(i) + '_S.png')
            save_image(D, folder_path + '/' + str(i) + '_D.png')

            save_image(D * (S / max), folder_path + '/' + str(i) + '_pN.png')
    folder_path = Savepath+'/model/'
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)

    torch.save(module.state_dict(), folder_path + str(epoch ) + '.pth')
    print('model===saved')




