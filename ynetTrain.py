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

from src.model.ynet import YNet

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import os
import random
import torch
from src.datahandler.denoise_dataset import DenoiseDataSet


path = '/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time6/HoloN/Img'

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





        noisyImage = torch.cat((noisy_img1,noisy_img2,noisy_img3),0)


        # 进行随机裁剪为256大小
        # 进行随机裁剪为256大小
        Hr = random.randint(0, 255)
        Wr = random.randint(0, 255)

        noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]


        return {'real_noisy1': noisyImage} # only noisy image dataset

class flowershow(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):


        dataset_path = os.path.join('/data/tmj/dataset/Utest/N')

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
        Hr = 255
        Wr = 255

        noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]

        # clean_img = self._load_img(file_name)

        # return {'clean': clean_img, 'real_noisy': noisy_img} # paired dataset
        return {'real_noisy1': noisyImage} # only noisy image dataset


saved_checkpoint = torch.load('/data/40/tmj/FakerFlower/RES/cdieYNetReS/model/58.pth')

depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
module = YNet().cuda()

# module.load_state_dict(saved_checkpoint,strict=True)


# LossS = nn.L1Loss(reduction='mean').cuda()
l1loss = nn.L1Loss(reduction='mean').cuda()

l2loss = nn.L1Loss(reduction='mean').cuda()
# op = optim.AdamW(module.parameters(),lr = 1e-4,weight_decay=1e-5)#定义优化器
train_data = flower256()
bs = 3
train_loader = DataLoader(train_data,batch_size=1,shuffle=False,drop_last=True)

start = 0
end = 40
Savepath = './RES/UandYformerCompare/Ynet2'

for epoch in range(start,end):
    module.train()

    for batch_id, x in enumerate(train_loader):

        noiseImg = x['real_noisy1'].cuda()
        D = x['D'].cuda()
        S = x['S'].cuda()


        op1 = optim.Adam(module.parameters(), lr=1e-4, weight_decay=1e-6)  # 定义优化器

        nowS, nowD = module(noiseImg)

        loss1 = l1loss(nowS, S) + l2loss(D,nowD)
        op1.zero_grad()
        loss1.backward()
        op1.step()

        print("epoch:   ",epoch, "loss1:",loss1.item())
        # print("epoch",epoch,"   loss2：",ssim_out2.data.item())

    if epoch >=0 :

        module.eval()
        testdata = flowershow()
        for i in range(95):

            x = testdata.__getitem__(i)['real_noisy1'].cuda()
            noise = x[1]
            folder_path = Savepath+'/picture/' + str(epoch)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)

            save_image(noise, folder_path + '/' + str(i) + '_N.png')
            x = x.unsqueeze(0)
            import time

            start_time = time.time()
            end_time = time.time()  # 记录结束时间

            execution_time = end_time - start_time

            S, D = module(x)
            print(f"执行时间：{execution_time}秒")


            save_image(S, folder_path + '/' + str(i) + '_S.png')
            save_image(D, folder_path + '/' + str(i) + '_D.png')

        print('pictureSaved')

        folder_path = Savepath+'/model/'
        if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)

        torch.save(module.state_dict(), folder_path + str(epoch ) + '.pth')
        print('model===saved')




