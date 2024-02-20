import torch
import torch.nn as nn

from OptimUtil import set_D
from model.DecoderFrame import DecoderFrame
from model.EncoderALL import encoderALL



class ModelFramePre(nn.Module):
    def __init__(self, netParams, hidden_number=16,channel=1):
        super(ModelFramePre, self).__init__()
        self.encoder = encoderALL(netParams,hidden_number=16,channel=channel)
        self.decoder = DecoderFrame(dd_in=channel)

    def forward(self, events_forward, events_backward, left_image, right_image, n_left, n_right):
        features = self.encoder(events_forward, events_backward, left_image, right_image, n_left, n_right)
        img = self.decoder(features)

        return img
if __name__ == '__main__':
    x = torch.randn(1,2,256,256,30).cuda()
    img1 = torch.randn(1,3,256,256).cuda()
    img2 = torch.randn(1,3, 256, 256).cuda()
    nb_of_time_bin = 15
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    model = ModelFramePre(netParams,channel=3,).cuda()

    import time
    start = time.time()
    print(model(x,x,img1,img2,torch.as_tensor(15).cuda().resize(1,),torch.as_tensor(15).resize(1,).cuda()).shape)
    end = time.time()
    print(end-start)