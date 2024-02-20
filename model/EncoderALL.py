import os

import torch.nn as nn
import torch
from model.EventImageFusion import EventImageFusion, Event_2_Image_2_Fusion, EventImageFusionLast
from model.EventEncoder import EventEncoder
from model.ImageEncoder import imageEncoder


class encoderALL(nn.Module):
    def __init__(self, netParams, imageSize = 256,hidden_number=16, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], channel=1, pretrain_ckpt=''):
        super(encoderALL, self).__init__()
        self.EventEncoder = EventEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.ImageEncoder = imageEncoder(embed_dim=hidden_number*2,dd_in=channel)
        self.Event_2_Image_2_Fusion0 = Event_2_Image_2_Fusion(64, imageSize//2)

        self.Event_2_Image_2_Fusion1 = Event_2_Image_2_Fusion(128,imageSize//4)

        self.Event_2_Image_2_Fusion2 = Event_2_Image_2_Fusion(256,imageSize//8)

        self.FeatureFusion_forward3 = EventImageFusion(1024,imageSize//16)
        self.FeatureFusion_backward3 = EventImageFusion(1024,imageSize//16)
        self.FeatureFusion_all3 = EventImageFusion(512,imageSize//16,True)

        self.FeatureFusionLast = EventImageFusionLast(256,imageSize//32)


    def forward(self, events_forward, events_backward, left_image, right_image, n_left, n_right):
        bs, _, H, W = left_image.shape
        [event_fea_forward_left, _] = self.EventEncoder(events_forward, n_left, n_right, True)
        [_, event_fea_backward_right] = self.EventEncoder(events_backward, n_left, n_right, False)


        img_fea_left = self.ImageEncoder(left_image,H,W)
        img_fea_right = self.ImageEncoder(right_image,H,W)
        img_fea_left_last = []
        img_fea_right_last = []
        fusion_fea_all= []

        for ii in range(4):
            img_fea_left_last.append( torch.cat((img_fea_left[ii],event_fea_forward_left[ii]),1))
            img_fea_right_last.append( torch.cat((img_fea_right[ii],event_fea_backward_right[ii]),1))


        fusion_fea_all.append(self.Event_2_Image_2_Fusion0(img_fea_left_last[0], img_fea_right_last[0]))
        fusion_fea_all.append(self.Event_2_Image_2_Fusion1(img_fea_left_last[1], img_fea_right_last[1]))
        fusion_fea_all.append(self.Event_2_Image_2_Fusion2(img_fea_left_last[2],img_fea_right_last[2]))



        fusion_fea_forward3 = self.FeatureFusion_forward3(img_fea_left_last[3])
        fusion_fea_backward3 = self.FeatureFusion_backward3(img_fea_right_last[3])


        fusion_fea_all.append( self.FeatureFusion_all3( torch.cat((fusion_fea_forward3,fusion_fea_backward3),1)))
        fusion_fea_all.append( self.FeatureFusionLast(fusion_fea_all[3]))


        return fusion_fea_all
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    h = 256
    w = 256
    x = torch.randn(1,2,h,w,4).cuda()
    img1 = torch.randn(1,1,h,w).cuda()
    img2 = torch.randn(1,1, h,w).cuda()
    nb_of_time_bin = 2
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    model = encoderALL(netParams,imageSize=h).cuda()
    torch.save(model.state_dict(), 'a.pth')
    import time
    start = time.time()
    model(x,x,img1,img2,torch.as_tensor(2).cuda().resize(1,),torch.as_tensor(2).resize(1,).cuda())
    end = time.time()
    print(end-start)