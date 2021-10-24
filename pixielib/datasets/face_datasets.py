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

### face training data
# cfg.dataset.head.scale_min = 1.4
# cfg.dataset.head.scale_max = 1.8
# cfg.dataset.head.trans_scale = 0.3
# cfg.dataset.head.blur_step = 1
from kornia.filters import median_blur, gaussian_blur2d, motion_blur
class VGGFace2(Dataset):
    def __init__(self, crop_size, scale=[1, 1], trans_scale = 0., blur_step=1, split='train'):
        '''
        K must be less than 6
        '''
        self.image_size = crop_size
        self.imagefolder = '/ps/scratch/face2d3d/train'
        self.kptfolder = '/ps/scratch/face2d3d/train_annotated_torch7'
        self.segfolder = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
        self.attfolder = '/ps/scratch/yfeng/Data/vggface2/gender-DEX'
        # hq:
        # datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        if split == 'train':
            datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_train_list_max_normal_100_ring_5_1_serial.npy'
        elif split == 'eval':
            datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_val_list_max_normal_100_ring_5_1_serial.npy'
            self.imagefolder = '/ps/scratch/face2d3d/test'
            self.kptfolder = '/ps/scratch/face2d3d/test_annotated_torch7'
        ## for test, no random crop
        if split != 'train':
            scale[0] = scale[1] = (scale[0] + scale[1])/2.
            trans_scale = 0.
        self.split = split
        self.data_lines = np.load(datafile).astype('str')
        self.cropper = array_cropper.Cropper(crop_size, scale, trans_scale)
        assert blur_step != 0
        self.blur_step = blur_step

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        # i = 0 # i: 0-6
        if self.split == 'train':
            i = np.random.randint(6)
        else:
            i = 5
        name = self.data_lines[idx, i]
        image_path = os.path.join(self.imagefolder, name + '.jpg')  
        seg_path = os.path.join(self.segfolder, name + '.npy')  
        kpt_path = os.path.join(self.kptfolder, name + '.npy')
        genderpath = os.path.join(self.attfolder, name + '_vote.txt')
        gender = self.load_gender(genderpath)

        image = imread(image_path)/255.
        kpt = np.load(kpt_path)[:,:2]
        mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

        ### crop information
        arrays = np.concatenate([image, mask], axis=-1)
        cropped_tensor, tform = self.cropper.crop(arrays, kpt)
        cropped_image = cropped_tensor[:,:,:3]
        cropped_mask = cropped_tensor[:,:,-1:]
        ## transform kpt 
        cropped_kpt = np.dot(np.hstack([kpt[:,:2], np.ones([kpt.shape[0],1])]), tform) # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
        cropped_kpt[:,[2]] = np.ones([kpt.shape[0],1]) # conf is 1
        
        ###
        image = torch.from_numpy(cropped_image.transpose(2,0,1)).type(dtype = torch.float32) 
        kpt = torch.from_numpy(cropped_kpt).type(dtype = torch.float32) 
        mask = torch.from_numpy(cropped_mask.transpose(2,0,1)).type(dtype = torch.float32) 

        image_hd = image.clone()

        if idx%self.blur_step  == 0:
            ### augment image resolution
            # blur_types = ['res', 'gaussian', 'median', 'motion']
            # more gaussian blur
            blur_types = ['res', 'gaussian', 'gaussian', 'motion']
            blur_type =  blur_types[np.random.randint(len(blur_types))]
            if blur_type == 'res':
                # self.resolution_scale : 0.1 ~ 1.2
                res_scale = np.random.rand() * 0.4 + 0.1
                image = image[None]
                res_size = int(res_scale*self.image_size)
                image = F.interpolate(image, (res_size, res_size), mode='bilinear')#, align_corners=False)
                image = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear')#, align_corners=False)
                image = image.squeeze()
            elif blur_type == 'gaussion':
                ks = np.random.randint(5)*2 + 9
                sigma = np.random.randint(5)*0.2 + 1.2
                image = image[None]
                image = util.gaussian_blur(image, kernel_size=(ks,ks), sigma=(sigma,sigma))
                image = image.squeeze()
            elif blur_type == 'median':
                ks = np.random.randint(3)*2 + 5
                image = image[None]
                image = median_blur(image, kernel_size=(ks,ks))
                image = image.squeeze()
            elif blur_type == 'motion':
                ks = np.random.randint(6)*2 + 7
                angle = np.random.rand()*180 - 90
                direction = np.random.randint(21)*0.1 - 1
                image = image[None]
                image = motion_blur(image, kernel_size=ks, angle=angle, direction=direction)
                image = image.squeeze()
                # for motion blur, the position will change
                image_hd = image
        ## 
        data_dict = {
            'image': image,
            'image_hd': image_hd,
            'face_kpt': kpt,
            'smplx_kpt': kpt,
            'mask': mask,
            'gender': gender
        }
        
        return data_dict

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask[:,:,None]

    def load_gender(self, genderpath):
        if os.path.exists(genderpath) is False:
            return 'N'
        with open(genderpath, 'r') as f:
            label = f.readline()
        if label == 'female' or label == 'Woman':
            return 'F'
        elif label == 'male' or label == 'Man':
            return 'M'
        else:
            return 'N'
