import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
from . import detectors

def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=2.0, face_detector='fan'):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp') + glob(testpath + '/*.jpeg')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]
        # print(imagepath)
        # exit()
        # import ipdb; ipdb.set_trace()
        image = np.array(cv2.imread(imagepath))[:,:,[2,1,0]]
        # image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            bbox = self.face_detector.run(image)
            if len(bbox) < 4:
                print(f'no face detected! run original image: {imagepath}')
                left = 0; right = h-1; top=0; bottom=w-1
            else:
                left = bbox[0]; right=bbox[2]
                top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'name': imagename,
                'imagepath': imagepath
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }


class NoWTest(Dataset):
    def __init__(self, iscrop=True, crop_size=224, scale=2.0, ):
        self.iscrop = iscrop
        self.scale = scale
        self.crop_size = crop_size
        # self.data_path = '/ps/scratch/face2d3d/texture_in_the_wild_code/NoW_validation/image_paths_ring_6_elements.npy'
        self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        self.data_lines = np.load(self.data_path).astype('str').flatten()
        # import ipdb; ipdb.set_trace()
        self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.resolution_inp = crop_size
    def __len__(self):
        return self.data_lines.shape[0]

    def bbox2point(self, left, right, top, bottom, type='type'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        image_th = []
        image_names =[]
        imagename = self.data_lines[index]
        imagepath = self.imagepath + self.data_lines[index] + '.jpg'
        bbx_path = self.bbxpath + self.data_lines[index] + '.npy'
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # bbox = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        # import ipdb; ipdb.set_trace()
        left = bbx_data['left']; right=bbx_data['right']
        top = bbx_data['top']; bottom=bbx_data['bottom']
        old_size, center = self.bbox2point(left, right, top, bottom, type='bbox')
        size = int(old_size*self.scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        
        return {'image': torch.tensor(dst_image).float(),
                'name': imagename,
                'imagepath': imagepath
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }


from . import detectors
class NoWTest_body(Dataset):
    def __init__(self, iscrop=False, crop_size=224, hd_size = 1024, scale=1.1, body_detector='rcnn', device='cuda:0'):
        self.iscrop = iscrop
        self.scale = scale
        self.crop_size = crop_size
        # self.data_path = '/ps/scratch/face2d3d/texture_in_the_wild_code/NoW_validation/image_paths_ring_6_elements.npy'
        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        self.data_path = '/ps/scratch/face2d3d/texture_in_the_wild_code/NoW_validation/image_paths._ring_6_elements.npy'
        self.data_lines = np.load(self.data_path).astype('str').flatten()
        # import ipdb; ipdb.set_trace()
        self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.resolution_inp = crop_size
        self.detector = detectors.FasterRCNN(device=device)
        self.crop_size = crop_size
        self.hd_size = hd_size
        self.scale = scale
        self.iscrop = iscrop

    def __len__(self):
        return self.data_lines.shape[0]

    def bbox2point(self, left, right, top, bottom, type='type'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        image_th = []
        image_names =[]
        imagename = self.data_lines[index]
        imagepath = self.imagepath + self.data_lines[index] + '.jpg'
        # bbx_path = self.bbxpath + self.data_lines[index] + '.npy'
        # image = np.array(imread(imagepath))
        # if len(image.shape) == 2:
        #     image = image[:,:,None].repeat(1,1,3)
        # if len(image.shape) == 3 and image.shape[2] > 3:
        #     image = image[:,:,:3]

        # h, w, _ = image.shape
        
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
        return {'image': torch.tensor(dst_image).float(),
                'name': imagename,
                'imagepath': imagepath, 
                'image_hd': torch.tensor(hd_image).float(),
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                'bbox': bbox
                }