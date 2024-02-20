import torch
import torch.nn as nn

from OptimUtil import set_D
from model.DecoderFlower import DecoderFlower
from model.EncoderALLFlower import encoderALL



class ModelFramePre(nn.Module):
    def __init__(self, netParams, hidden_number=16,channel=3):
        super(ModelFramePre, self).__init__()
        self.encoder = encoderALL(netParams,hidden_number=16,channel=channel)
        self.decoder = DecoderFlower(dd_in=1)

    def forward(self, events_forward, events_backward,img ,n_left, n_right):
        features = self.encoder(events_forward, events_backward, img, n_left, n_right)
        S,D = self.decoder(features)

        return S,D
# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#     x = torch.randn(1,2,256,256,4).cuda()
#     img1 = torch.randn(1,3,256,256).cuda()
#
#     nb_of_time_bin =2
#     netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
#     model = ModelFramePre(netParams,channel=3,).cuda()
#
#     set_D(model)
#
#     S,D = model(x, x, img1, torch.as_tensor(2).cuda().resize(1, ), torch.as_tensor(2).resize(1, ).cuda())


