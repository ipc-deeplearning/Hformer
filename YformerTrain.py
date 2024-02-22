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
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import os
import random
import torch
from src.datahandler.denoise_dataset2 import DenoiseDataSet
len = 500
path = '/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time7/HoloN/Img'
testpath = '/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time7/HoloN/Img'
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

        dataset_path = os.path.join(testpath)

        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

        for root, _, files in os.walk("/data/40/ntmj/ntmj/eventMain/data/YnetTrain/Time7/HoloD/Img"):
            for file_name in files:
                self.Dimg.append(os.path.join(root, file_name))

        self.img_paths.sort()

        self.Dimg.sort()


    def _load_data(self, data_idx):

        file_name1 = self.img_paths[data_idx]
        noisy_img1 = self._load_img(file_name1)

        file_name2 = self.img_paths[data_idx+1]
        noisy_img2 = self._load_img(file_name2)

        file_name3 = self.img_paths[data_idx+2]
        noisy_img3 = self._load_img(file_name3)

        noisyImage = torch.cat((noisy_img1,noisy_img2,noisy_img3),0)


        Dp = self.Dimg[data_idx+1]
        D = self._load_img(Dp)


        # # 进行随机裁剪为256大小
        # Hr = random.randint(0, 767)
        # Wr = random.randint(0,767)
        #
        # noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]


        # clean_img = self._load_img(file_name)
        Hr = 30
        Wr = 30

        noisyImage = noisyImage[:,Hr:Hr+256,Wr:Wr+256]
        D = D[:,Hr:Hr+256, Wr:Wr+256]



        # return {'clean': clean_img, 'real_noisy': noisy_img} # paired dataset
        return {'real_noisy1': noisyImage,"gt":D} # only noisy image dataset


saved_checkpoint = torch.load('/data/40/ntmj/ntmj/eventMain/out/YformerRes114/model/39.pth')
# saved_checkpoint = torch.load('/data/tmj/FakerFlower/RES/NewShipingFlower/model/35.pth')
# saved_checkpoint = torch.load('/data/tmj/FakerFlower/test2/noPre2/model99.pth')

module = UformerY(img_size=256,embed_dim=32).cuda()
module = module.cuda()


# module.load_state_dict(torch.load("/data/40/tmj/FakerFlower/reallyBest.pth"),strict=True)
# moduleB.load_state_dict(saved_checkpointB,strict=True)

# LossS = nn.L1Loss(reduction='mean').cuda()
l1loss = nn.L1Loss(reduction='mean').cuda()

# l1loss = nn.CrossEntropyLoss()
# op = optim.AdamW(module.parameters(),lr = 1e-4,weight_decay=1e-5)#定义优化器
train_data = flower256()
bs = 3
train_loader = DataLoader(train_data,batch_size=1,shuffle=False,drop_last=True)


ssim_loss = SSIM().cuda()

start = 40
end = 100
flodpa = '/data/40/ntmj/ntmj/eventMain/out/YformerRes114/'
bestloss = 99999

op1 = optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)  # 定义优化器
import time

for epoch in range(start,end):
    module.train()

    start_time = time.time()
    for batch_id, x in enumerate(train_loader):
        noiseImg = x['real_noisy1'].cuda()
        label = x['real_noisy2'].cuda()
        parmS = set_S_down(module)
        op1 = optim.Adam(parmS, lr=1e-4, weight_decay=1e-6)  # 定义优化器

        preS1 = label[:, 0, :, :].unsqueeze(1)

        nowS, nowD = module(noiseImg)
        loss1 = l1loss(nowS, preS1)
        op1.zero_grad()
        loss1.backward()
        op1.step()

        parm_D = set_D(module)
        op3 = optim.Adam(parm_D, lr=1e-4, weight_decay=1e-6)  # 定义优化器
        nowS, nowD = module(noiseImg)

        cc = torch.reshape(nowS, (1, -1))
        max = torch.mode(cc)[0]
        pt = noiseImg[:, 1, :, :].unsqueeze(1)

        ssim_out2 = 1 - ssim_loss(pt, nowD * (nowS / max))
        op3.zero_grad()
        ssim_out2.backward(retain_graph=False)
        op3.step()

        print("epoch:   ",epoch, "loss1:",loss1.item()," loss3",ssim_out2.data.item())
        # print("epoch",epoch,"   loss2：",ssim_out2.data.item())
    end_time = time.time()
    print(end_time-start_time)

    if epoch >=30 and epoch%2==0:

        module.eval()
        testdata = flowershow()

        for i in range(60):
            x = testdata.__getitem__(i)['real_noisy1'].cuda()

            x = x.unsqueeze(0)

            nowS, nowD = module(x)

            cc = torch.reshape(nowS, (1, -1))
            max = torch.mode(cc)[0]
            pt = x[:, 1, :, :].unsqueeze(1)
            #
            # # loss1 = l1loss(nowS, S) +  1 - l2loss(D,nowD)
            # # op1.zero_grad()
            # # loss1.backward()
            # # op1.step()
            #
            Savepath = flodpa+"/"+ str(epoch )+"./"
            if not os.path.exists(Savepath):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(Savepath)
            save_image(nowD * (nowS / max), Savepath + '/' + str(i) + '_pN.png')
            save_image(nowS, Savepath + '/' + str(i) + '_S.png')
            save_image(nowD, Savepath + '/' + str(i) + '_D.png')


    folder_path = flodpa+'/model/'
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)

    torch.save(module.state_dict(), folder_path + str(epoch ) + '.pth')
    print('model===saved')




