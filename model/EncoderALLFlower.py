import os

import torch.nn as nn
import torch
from model.EventImageFusionFlower import EventImageFusion, Event_2_Image_2_Fusion, EventImageFusionLast
from model.EventEncoderFlower import EventEncoder
from model.ImageEncoder import imageEncoder


class encoderALL(nn.Module):
    def __init__(self, netParams, imageSize = 256,hidden_number=16, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], channel=1, pretrain_ckpt=''):
        super(encoderALL, self).__init__()
        self.EventEncoder = EventEncoder(netParams, hidden_number//2, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)


        self.ImageEncoder = imageEncoder(embed_dim=hidden_number*2,dd_in=channel)

        self.Event_2_Image_2_Fusion0 = Event_2_Image_2_Fusion(64, imageSize//2)

        self.Event_2_Image_2_Fusion1 = Event_2_Image_2_Fusion(128,imageSize//4)

        self.Event_2_Image_2_Fusion2 = Event_2_Image_2_Fusion(256,imageSize//8)

        self.Event_2_Image_2_Fusion3 = Event_2_Image_2_Fusion(512,imageSize//16)

    def forward(self, events_forward,events_back, imgs, n_left, n_right):
        bs, _, H, W = imgs.shape
        [event_fea_forward_left, _] = self.EventEncoder(events_forward, n_left, n_right, True)

        [_, event_fea_backward_right] = self.EventEncoder(events_back, n_left, n_right, False)

        for ij in range(4):
            event_fea_forward_left[ij] = torch.cat((event_fea_forward_left[ij],event_fea_backward_right[ij]),1)


        img_fea_left = self.ImageEncoder(imgs,H,W)

        img_fea_left_last = []

        fusion_fea_all= []

        for ii in range(4):
            img_fea_left_last.append( torch.cat((img_fea_left[ii],event_fea_forward_left[ii]),1))



        fusion_fea_all.append(self.Event_2_Image_2_Fusion0(img_fea_left_last[0] ))
        fusion_fea_all.append(self.Event_2_Image_2_Fusion1(img_fea_left_last[1]))
        fusion_fea_all.append(self.Event_2_Image_2_Fusion2(img_fea_left_last[2]))


        fusion_fea_all.append( self.Event_2_Image_2_Fusion3(img_fea_left_last[3]))





        return fusion_fea_all
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    h = 256
    w = 256
    x = torch.randn(1,2,h,w,4).cuda()
    img1 = torch.randn(1,3,h,w).cuda()

    nb_of_time_bin = 2
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    model = encoderALL(netParams,imageSize=h,channel=3).cuda()



    import time
    start = time.time()
    model(x,x,img1,torch.as_tensor(2).cuda().resize(1,),torch.as_tensor(2).resize(1,).cuda())
    end = time.time()
    print(end-start)