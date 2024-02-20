import os
import sys

from torchvision.utils import save_image

from OptimUtil import set_S_down, set_D
from model.ModelFlower import ModelFramePre

import torch.nn.functional as F
import random
from tools import hybrid_storage, representation
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from tools.ssimLoss import SSIM

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


def calssim(gt, pred):
    return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=False, gaussian_weights=True)


nums_frame = 48
nums_lowRgb = 3
total_frame = 1
imgSize = 511
length = 600

folder_path = "./reallEventShow"
if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(folder_path)


class UHSEDataset:
    def __init__(self, p, nb_of_timebin=5):
        folder = p.strip('\n')
        self.storage = [hybrid_storage.HybridStorage.from_folders(event_folder=folder,
                                                                  gt_image_folder=os.path.join(folder, 'frame'),
                                                                  image_file_template="*.png",
                                                                  gt_img_timestamps_file='../timestamps.txt',
                                                                  event_name='event.npy')]


        self.idx = []
        for k in range(len(self.storage)):
            self.idx += [k] * (nums_lowRgb*nums_frame)
        self.start_idx = [k*nums_lowRgb*nums_frame for k in range(len(self.storage))]
        self.nb_of_time_bin = nb_of_timebin
        self.name = os.path.join(os.path.split(folder)[-1][:-1], os.path.split(folder)[-1])

    def __len__(self):
        # length = 0
        # for k in range(len(self.storage)):
        #     length += nums_lowRgb*nums_frame
        return length-2

    def __getitem__(self, idx1):
        sample_idx = 0

        # sample_idx = self.idx[idx1]
        # start_idx = self.start_idx[sample_idx]

        idx0 = idx1 +1

        idx = idx0

        t = self.storage[sample_idx]._gtImages._timestamps[idx]

        idx_r =idx0 + 1
        idx_l = idx0 - 1

        # left_image = self.storage[sample_idx]._gtImages._images[idx_l]
        # right_image = self.storage[sample_idx]._gtImages._images[idx_r]

        t_left = self.storage[sample_idx]._gtImages._timestamps[idx_l]
        t_right = self.storage[sample_idx]._gtImages._timestamps[idx_r]

        input_Img = self.storage[sample_idx]._gtImages._images[idx]

        input_Img0 = self.storage[sample_idx]._gtImages._images[idx_l]
        input_Img2 = self.storage[sample_idx]._gtImages._images[idx_r]



        duration_left = t-t_left
        duration_right = t_right-t

        event_left = self.storage[sample_idx]._events.filter_by_timestamp(t_left, duration_left)
        event_right = self.storage[sample_idx]._events.filter_by_timestamp(t, duration_right)

        image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # img1 = event_left.to_image()


        for x,y,t,p in event_left._features:
            color=  [255, 0, 0] if p == -1 else [0, 0, 255]
            x,y = int(x),int(y)
            image[x,y]=color

        cv2.imwrite(folder_path+str(idx)+"L.png", image)



        # for x,y,t,p in event_right._features:
        #     color=  [255, 0, 0] if p == -1 else [0, 0, 255]
        #     x,y = int(x),int(y)
        #     image[x,y]=color
        #
        # cv2.imwrite(folder_path+str(idx)+"R.png", image)








        n_left = n_right = self.nb_of_time_bin
        event_left_forward = representation.to_count_map(event_left, n_left).clone()
        event_right_forward = representation.to_count_map(event_right, n_right).clone()



        event_right.reverse()
        event_left.reverse()
        event_left_backward = representation.to_count_map(event_left, n_left)
        event_right_backward = representation.to_count_map(event_right, n_right)
        events_forward = np.concatenate((event_left_forward, event_right_forward), axis=-1)
        events_backward = np.concatenate((event_right_backward, event_left_backward), axis=-1)
        #
        # events_forward = np.squeeze(events_forward, -1)
        #
        # events_backward = np.squeeze(events_backward, -1)

        input_Img = torch.cat((input_Img0,input_Img,input_Img2),0)


        # # 进行随机裁剪为256大小
        # Hr = random.randint(0, imgSize-256-1)
        # Wr = random.randint(0,imgSize-256-1)
        #
        # input_Img = input_Img[:,Hr:Hr+256,Wr:Wr+256]
        #
        # gt_image = gt_image[:,Hr:Hr+256,Wr:Wr+256]
        #
        # events_forward = events_forward[:,Hr:Hr+256,Wr:Wr+256,:]
        # events_backward = events_backward[:,Hr:Hr+256,Wr:Wr+256,:]


        return  input_Img, idx,events_forward,events_backward




def showMessage(message, file):
    print(message)
    with open(file, 'a') as f:
        f.writelines(message + '\n')


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    cv2.imwrite(path, np.array(img[0, 0].cpu() * 255))


if __name__ == '__main__':
    folder_path = './out/pictureshow/ReallyRes/150Test'


    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)




    import torch
    from torch import nn, optim
    split_by_scenario = False
    ckpt_path = '/data/40/ntmj/ntmj/eventMain/out/time512ReallyLD/model/150.pth'
    data_path = '/data/40/ntmj/ntmj/eventMain/data/event512'



    input_img_channel = 1
    nb_of_time_bin = 2
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}



    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    # 传入 fast——ynthesisModule的结果权重
    model = ModelFramePre(netParams, hidden_number=16)







    if not split_by_scenario:
        with open(os.path.join(data_path, 'trainreally.txt'), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(data_path, 'testreally.txt'), 'r') as f:
            linesTest = f.readlines()
    else:
        with open(os.path.join(data_path, 'test.txt'), 'r') as f:
            lines = f.readlines()



    if ckpt_path!='':
        model.load_state_dict(torch.load(ckpt_path))

        print('==> loading existing model:', ckpt_path)


    op3 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)  # 定义优化器


    train_loader = [UHSEDataset(lines[k], nb_of_timebin=nb_of_time_bin) for k in range(len(lines))]
    train_loader = [torch.utils.data.DataLoader(train_loader[k], batch_size=1, shuffle=False, pin_memory=True, num_workers=1) for k in range(len(lines))]

    test_loader = [UHSEDataset(linesTest[k], nb_of_timebin=nb_of_time_bin) for k in range(len(linesTest))]
    test_loader = [ torch.utils.data.DataLoader(test_loader[k], batch_size=1, shuffle=False, pin_memory=True, num_workers=1) for k in range(len(linesTest))]

    model = model.cuda()




    ssim_loss = SSIM().cuda()

    l1loss = nn.L1Loss(reduction='mean').cuda()

    num = 512 / 128 - 1
    r = torch.randn(512, 512)

    # 设置裁剪参数
    image_width, image_height = 512, 512
    crop_size = 256
    overlap = 128

    # 计算裁剪次数和步进值
    crop_count_x = (image_width - overlap) // (crop_size - overlap)
    crop_count_y = (image_height - overlap) // (crop_size - overlap)




    with torch.no_grad():
        model.eval()
        psnr, ssim = [], []

        for loader in test_loader:
            count = 0

            for i, (input_Img, idx,events_forward,events_backward )in enumerate(loader):


                # x = torch.randn(1, 2, 256, 256, 4).cuda()
                # img1 = torch.randn(1, 3, 256, 256).cuda()

                x = input_Img.cuda()
                events_forward = events_forward.cuda()

                events_backward = events_backward.cuda()

                # 开始裁剪、命名、输入到网络和拼接
                # for j1 in range(crop_count_x):
                #     for j2 in range(crop_count_y):
                #         left = j1 * (crop_size - overlap)
                #         upper = j2 * (crop_size - overlap)
                #         right = left + crop_size
                #         lower = upper + crop_size
                #         cropped_image = x[:, :, left:right, upper:lower]
                #
                #         events_forward_cropped = events_forward[:, :, left:right, upper:lower]
                #         events_backward_cropped = events_backward[:, :, left:right, upper:lower]
                #         # 转换图像为张量
                #
                #         # 输入图像到模型
                #         with torch.no_grad():
                #             nowS, I = model(events_forward_cropped, events_backward_cropped, cropped_image,
                #                                torch.as_tensor(2).cuda().view(1, ),
                #                                torch.as_tensor(2).view(1, ).cuda())
                #             # I = cropped_image
                #             I = I.squeeze()
                #         if j1 == 0:
                #             if j2 == 0:
                #                 r[0:192, 0:192] = I[0:192, 0:192]
                #             elif j2 == num - 1:
                #                 r[0:192, 320:512] = I[0:192, 64:256]
                #             else:
                #                 r[0:192, j2 * 128 + 64:(j2 + 1) * 128 + 64] = I[0:192, 64:192]
                #         elif j1 == num - 1:
                #             if j2 == 0:
                #                 r[320:512, 0:192] = I[64:256, 0:192]
                #             elif j2 == num - 1:
                #                 r[320:512, 320:512] = I[64:256, 64:256]
                #             else:
                #                 r[320:512, j2 * 128 + 64:(j2 + 1) * 128 + 64] = I[64:256, 64:192]
                #         else:
                #             if j2 == 0:
                #                 r[j1 * 128 + 64:(j1 + 1) * 128 + 64, 0:192] = I[64:192, 0:192]
                #             elif j2 == num - 1:
                #                 r[j1 * 128 + 64:(j1 + 1) * 128 + 64, 320:512] = I[64:192, 64:256]
                #             else:
                #                 r[j1 * 128 + 64:(j1 + 1) * 128 + 64, j2 * 128 + 64:(j2 + 1) * 128 + 64] = I[64:192,
                #                                                                                           64:192]




                # save_image(nowS, folder_path+ '/predictD' + str(i+2) + '_S.png')
                save_image(r, folder_path+ '/pD' +'{:03d}.png'.format(i+2))
                print("picturesave:"+str(i))

                # save_image(input_Img[:,1,:,:], folder_path + '/' + str(i) + '_N.png')



