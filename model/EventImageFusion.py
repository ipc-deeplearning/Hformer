import torch
import torch.nn as nn
import torch.nn.functional as F

from model.UformerY import BasicUformerLayer

#512 16
class EventImageFusion(nn.Module):
    def __init__(self,embed_dim,img_size,last=False):
        super(EventImageFusion, self).__init__()

        # Encoder
        if last ==True:
            self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1)
            self.fusion1 = BasicUformerLayer(dim=embed_dim // 2, output_dim=embed_dim // 2,
                                             input_resolution=(img_size, img_size), depth=2,
                                             num_heads=4, win_size=8)
        else:
            self.conv1 = nn.Conv2d(embed_dim, embed_dim // 4, 3, padding=1)
            self.fusion1 = BasicUformerLayer(dim=embed_dim//4, output_dim=embed_dim//2, input_resolution=(img_size,img_size), depth=2,
                                                    num_heads=4, win_size=8)


    def forward(self, image_event_fea):
        B,C,H,W = image_event_fea.shape
        output = self.conv1(image_event_fea)

        output=output.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        output = self.fusion1(output)
        output = output.transpose(1, 2).contiguous().view(B, -1, H, W)

        return output

class Event_2_Image_2_Fusion(nn.Module):
    def __init__(self,embed_dim,img_size):
        super(Event_2_Image_2_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim*2,embed_dim//4,3,padding=1)
        self.conv2 = nn.Conv2d(embed_dim*2,embed_dim//4,3,padding=1)

        # Encoder
        self.fusion1 = BasicUformerLayer(dim=embed_dim//2, output_dim=embed_dim//4, input_resolution=(img_size,img_size), depth=1,
                                                num_heads=4, win_size=8)


    def forward(self, image_event_fea,event_fea):
        B,C,H,W = image_event_fea.shape
        output1 = self.conv1(image_event_fea)
        output2 = self.conv1(event_fea)
        output = torch.concat((output1,output2),1)

        output=output.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        output = self.fusion1(output)
        output = output.transpose(1, 2).contiguous().view(B, -1, H, W)

        return output

class EventImageFusionLast(nn.Module):
    def __init__(self,embed_dim=256,img_size=8):
        super(EventImageFusionLast, self).__init__()

        self.conv1 = nn.Conv2d(embed_dim, embed_dim*2,kernel_size=4, stride=2, padding=1)
        if img_size<8:
            winsize = img_size
        else:
            winsize = 8
        self.fusion1 = BasicUformerLayer(dim=embed_dim*2, output_dim=embed_dim*2, input_resolution=(img_size,img_size), depth=2,
                                                num_heads=4, win_size=winsize)


    def forward(self, image_event_fea):

        output = self.conv1(image_event_fea)
        B, C, H, W = output.shape
        output=output.flatten(2).transpose(1, 2).contiguous()  # B H*W C

        output = self.fusion1(output)
        output = output.transpose(1, 2).contiguous().view(B, -1, H, W)

        return output