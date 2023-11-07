import os
import torch
from torchvision.utils import save_image
from src.datahandler.YnetReadFlower import flowershow
from src.model.UformerY4 import UformerY

from  src.model.ynet import YNet

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

module = YNet().cuda()

module.load_state_dict(torch.load('/data/tmj/FakerFlower/RES/ynet/model/8.pth'))
module.eval()
testdata = flowershow()

import time

# 使用time()测量程序运行时间
start_time = time.time()
for i in range(50):
    x = testdata.__getitem__(i)['real_noisy1'].cuda()

    noise1 = x[1]

    # folder_path = '/data/tmj/ynetRes/D'
    # if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(folder_path)



    x = x.unsqueeze(0)
    S, D = module(x)
    # cc = torch.reshape(S, (1, -1))
    # max = torch.mode(cc)[0]
    #
    # save_image(S/max, folder_path + '/' + str(i) + '_Rs.png')
    # save_image(S, folder_path + '/' + str(i) + '.png')
    # save_image(D, folder_path + '/' + str(i) + '.png')


# 待测程序
end_time = time.time()
time_cost = end_time - start_time
print("time cost:", time_cost)
