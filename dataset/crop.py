import cv2
import os
# for i in sorted(os.listdir("/data/tmj/LGBnet/dataset/160_105")):
#     img = cv2.imread(i)
#     cropped = cv2.resize(img,)
#     cv2.imwrite("./NEW50/crop"+i,cropped)
dataset_path = os.path.join('/data/tmj/LGBnet/dataset/160_105')
# dataset_path = os.path.join('/data/tmj/LGBnet/dataset/old_crop')

assert os.path.exists(dataset_path), 'There is no dataset %s' % dataset_path
i = 0
for root, _, files in os.walk(dataset_path):
    for file_name in files:
        o = os.path.join(root, file_name)
        img = cv2.imread(o)
        cropped = cv2.resize(img,(256,256))
        cv2.imwrite("/data/tmj/LGBnet/dataset/160_105_resize256/" + file_name, cropped)
        i = i+1

print( i )
