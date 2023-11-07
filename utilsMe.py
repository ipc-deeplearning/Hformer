import cv2
import numpy as np
import matplotlib.pyplot as plt

def spreate(N1,N2,S,D,v1,v2):
    lk_params = dict(winSize=(15, 15),  # 搜索窗口的大小
                     maxLevel=2,  # 图像金字塔的最大层数
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # 计算光流
    N1 = N1.cpu().detach().numpy() * 255
    N2 = N2.cpu().detach().numpy() * 255
    D = D.cpu().detach().numpy() * 255
    S = S.cpu().detach().numpy() * 255
    flow = cv2.calcOpticalFlowFarneback(N1, N2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    thre, img = cv2.threshold(flow_magnitude, (v1+v2+1)/2, 255, cv2.THRESH_BINARY)  # 二值
    fimg = 255 - img

    total = np.sum(N2)
    wd = np.sum(N2 * img / 255.0)
    ws = np.sum(N2 * fimg / 255.0)

    kernel = np.ones((5, 5), np.uint8)  # 这里使用了一个5x5的矩形内核
    # 执行开运算
    opening_result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) / 255.0
    # 执行开运算
    close = cv2.morphologyEx(fimg, cv2.MORPH_CLOSE, kernel) / 255.0
    Dc = close * D
    Sc = close * S
    D = D * opening_result
    S = S * opening_result
    # 创建一个掩码，将像素值为0的区域标记为False，其余区域标记为True
    mask = (D != 0)
    # 将掩码应用到图像，将像素值为0的区域置为0
    image_without_zeros = np.where(mask, D, np.nan)
    # 计算非零像素的平均值
    mean_value = np.nanmean(image_without_zeros)
    # 计算每个非零像素与平均值的差异
    diff = image_without_zeros - mean_value
    # 计算方差
    d1 = np.nanmean(diff ** 2)
    if np.isnan(d1):
        return 0

    # 创建一个掩码，将像素值为0的区域标记为alse，其余区域标记为True
    maskS = (S != 0)
    # 将掩码应用到图像，将像素值为0的区域置为0
    image_without_zeros = np.where(maskS, S, np.nan)
    # 计算非零像素的平均值
    mean_value = np.nanmean(image_without_zeros)
    # 计算每个非零像素与平均值的差异
    diff = image_without_zeros - mean_value
    # 计算方差
    s1 = np.nanmean(diff ** 2)
    mask = (Dc != 0)
    # 将掩码应用到图像，将像素值为0的区域置为0
    image_without_zeros = np.where(mask, D, np.nan)
    # 计算非零像素的平均值
    mean_value = np.nanmean(image_without_zeros)
    # 计算每个非零像素与平均值的差异
    diff = image_without_zeros - mean_value
    # 计算方差
    d2 = np.nanmean(diff ** 2)
    if np.isnan(d2):
        return 0

    # 创建一个掩码，将像素值为0的区域标记为False，其余区域标记为True
    maskS = (Sc != 0)
    # 将掩码应用到图像，将像素值为0的区域置为0
    image_without_zeros = np.where(maskS, S, np.nan)
    # 计算非零像素的平均值
    mean_value = np.nanmean(image_without_zeros)
    # 计算每个非零像素与平均值的差异
    diff = image_without_zeros - mean_value
    # 计算方差
    s2 = np.nanmean(diff ** 2)
    Dad = d1 / d2
    Sas = s2 / s1
    res = ws / total * Sas + wd / total * Dad
    if np.isnan(res):
        return 0

    return res

def iou(a, b, epsilon=1e-5):
    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)
    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou

def calculate_iou(mask1, mask2):
    # 将二值掩码转换为布尔值掩码
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    # 计算交集部分的布尔值掩码
    intersection = np.logical_and(mask1_bool, mask2_bool)

    # 计算并集部分的布尔值掩码
    union = np.logical_or(mask1_bool, mask2_bool)

    # 计算交集和并集的像素数量
    intersection_pixels = np.count_nonzero(intersection)
    union_pixels = np.count_nonzero(union)

    # 计算IoU
    iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0

    return iou
def cluIoU(N1,N2,S,D,glN):
    lk_params = dict(winSize=(15, 15),  # 搜索窗口的大小
                     maxLevel=2,  # 图像金字塔的最大层数
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # 计算光流
    S = S.squeeze()
    D = D.squeeze()
    N1 = N1.cpu().detach().numpy() * 255
    N2 = N2.cpu().detach().numpy() * 255
    D = D.cpu().detach().numpy() * 255
    S = S.cpu().detach().numpy() * 255
    flow = cv2.calcOpticalFlowFarneback(N1, N2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    thre, mask = cv2.threshold(flow_magnitude, glN, 255, cv2.THRESH_BINARY)  # 二值

    if mask.any() == 0:
        return 0,0

    kernel = np.ones((11, 11), np.uint8)  # 这里使用了一个5x5的矩形内核

    S = cv2.convertScaleAbs(S)
    D = cv2.convertScaleAbs(D)
    N2 = cv2.convertScaleAbs(N2)



    ips, S = cv2.threshold(S, 0, 255, cv2.THRESH_OTSU)
    ipd, D = cv2.threshold(D, 0, 255, cv2.THRESH_OTSU)

    ip2, n2 = cv2.threshold(N2, 0, 255, cv2.THRESH_OTSU)

    n2 = 255 - n2
    Dmask = mask * n2

    Dmask = cv2.morphologyEx(Dmask, cv2.MORPH_CLOSE, kernel)
    Dmask = cv2.morphologyEx(Dmask, cv2.MORPH_OPEN, kernel)



    Smask = n2 - Dmask

    if Smask.any() == 0:
        return 0,0

    if Dmask.any() == 0:
        return 0,0


    Smask = cv2.morphologyEx(Smask, cv2.MORPH_CLOSE, kernel)
    Smask = cv2.morphologyEx(Smask, cv2.MORPH_OPEN, kernel)

    D = cv2.morphologyEx(D, cv2.MORPH_CLOSE, kernel)
    D = cv2.morphologyEx(D, cv2.MORPH_OPEN, kernel)

    # 执行开运算

    S = cv2.morphologyEx(S, cv2.MORPH_CLOSE, kernel)
    S = cv2.morphologyEx(S, cv2.MORPH_OPEN, kernel)
    D = 255-D
    S = 255-S

    return iou(D,Dmask),iou(S,Smask)

