import os, sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from ..utils import util
from . import detectors
from ..utils import array_cropper

def build_dataloader(testpath, batch_size=1):
    data_list = []
    dataset = TestData(testpath = testpath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last =False)
    return dataset, dataloader

def video2sequence(video_path):
    print('extract frames from video: {}...'.format(video_path))
    videofolder = video_path.split('.')[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        success,image = vidcap.read()
        if image is None:
            break
        if count % 1 == 0:
            imagepath = '{}/{}_frame{:05d}.jpg'.format(videofolder, video_name, count)
            cv2.imwrite(imagepath, image)     # save frame as JPEG file
            imagepath_list.append(imagepath)
        count += 1

    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=False, crop_size=224, hd_size = 1024, scale=1.1, body_detector='rcnn', device='cpu'):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.jpeg')
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'MOV']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the input path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.hd_size = hd_size
        self.scale = scale
        self.iscrop = iscrop
        if body_detector == 'rcnn':
            self.detector = detectors.FasterRCNN(device=device)
        elif body_detector == 'keypoint':
            self.detector = detectors.KeypointRCNN(device=device)
        elif body_detector == 'mmdet':
            self.detector = detectors.MMDetection(device=device)
        else:
            print('no detector is used')

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:,:,:3]/255.
        h, w, _ = image.shape

        image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)[None, ...]
        if self.iscrop:
            bbox = self.detector.run(image_tensor)
            if bbox is None:
                print('no person detected! run original image')
                left = 0; right = w-1; top=0; bottom=h-1
            else:
                left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
            left = 0; right = w-1; top=0; bottom=h-1
            bbox = [left, top, right, bottom]

        # crop image
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2,0,1)
        # hd image
        DST_PTS = np.array([[0,0], [0,self.hd_size - 1], [self.hd_size - 1, 0]])
        tform_hd = estimate_transform('similarity', src_pts, DST_PTS)
        hd_image = warp(image, tform_hd.inverse, output_shape=(self.hd_size, self.hd_size))
        hd_image = hd_image.transpose(2,0,1)
        # crop image
        return {'image': torch.tensor(dst_image).float(),
                'name': imagename,
                'imagepath': imagepath,
                'image_hd': torch.tensor(hd_image).float(),
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                'bbox': bbox,
                'size': size,
                }
