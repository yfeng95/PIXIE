import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixie import PIXIE
# from pixielib.pixie_parallel import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils.tensor_cropper import transform_points

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    # check env
    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # load test images 
    testdata = TestData(args.inputpath, iscrop=args.iscrop, body_detector='rcnn')

    #-- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)
    if args.deca_path:
        # if given deca code path, run deca to get face details, here init deca model
        sys.path.insert(0, args.deca_path)
        from decalib.deca import DECA
        deca = DECA(device=device)
        use_deca = True
    else:
        use_deca = False
    
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        util.move_dict_to_device(batch, device)
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        # print(name)
        # frame_id = int(name.split('frame')[-1])
        # name = f'{frame_id:05}'

        data = {
            'body': batch
        }
        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
        # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
        moderator_weight = param_dict['moderator_weight']
        codedict = param_dict['body']
        opdict = pixie.decode(codedict, param_type='body')
        opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
        if args.saveObj or args.saveParam or args.savePred or args.saveImages or args.deca_path is not None:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        # run deca if deca is available and moderator thinks information from face crops is reliable
        if args.deca_path is not None and param_dict['moderator_weight']['head'][0,1].item()>0.6:
            cropped_face_savepath = os.path.join(savefolder, name, f'{name}_facecrop.jpg')
            cv2.imwrite(cropped_face_savepath, util.tensor2image(data['body']['head_image'][0]))
            _, deca_opdict, _ = deca.run(cropped_face_savepath)
            flame_displacement_map = deca_opdict['displacement_map']
            opdict['displacement_map'] = visualizer.tex_flame2smplx(flame_displacement_map)
        if args.lightTex:
            visualizer.light_albedo(opdict)
        if args.extractTex:
            visualizer.extract_texture(opdict, data['body']['image_hd'])
        if args.reproject_mesh and args.rasterizer_type=='standard':
            ## whether to reproject mesh to original image space
            tform = batch['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2)
            original_image = batch['original_image'][None, ...]
            visualizer.recover_position(opdict, batch, tform, original_image)
        if args.saveVis:
            if args.showWeight is False:
                moderator_weight = None 
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, moderator_weight=moderator_weight, use_deca=use_deca)
            # show cropped parts 
            if args.showParts:
                visdict['head'] = data['body']['head_image']
                visdict['left_hand'] = data['body']['left_hand_image'] # should be flipped
                visdict['right_hand'] = data['body']['right_hand_image']
            cv2.imwrite(os.path.join(savefolder, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=args.render_size))
            # print(os.path.join(savefolder, f'{name}_vis.jpg'))
            # import ipdb; ipdb.set_trace()
            # exit()
        if args.saveGif:
            visualizer.rotate_results(opdict, visdict=visdict, savepath=os.path.join(savefolder, f'{name}_vis.gif'))
        if args.saveObj:
            visualizer.save_obj(os.path.join(savefolder, name, f'{name}.obj'), opdict)
        if args.saveParam:
            codedict['bbox'] = batch['bbox']
            util.save_pkl(os.path.join(savefolder, name, f'{name}_param.pkl'), codedict)
            np.savetxt(os.path.join(savefolder, name, f'{name}_bbox.txt'), batch['bbox'].squeeze())
        if args.savePred:
            util.save_pkl(os.path.join(savefolder, name, f'{name}_prediction.pkl'), opdict) 
        if args.saveImages:
            for vis_name in visdict.keys():
                cv2.imwrite(os.path.join(savefolder, name, f'{name}_{vis_name}.jpg'), util.tensor2image(visdict[vis_name][0]))
                       
    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    parser.add_argument('-i', '--inputpath', default='TestSamples/body', type=str,
                        help='path to the test data, can be image folder, image path, image path list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/body/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    # rendering option
    parser.add_argument('--render_size', default=1024, type=int,
                        help='image size of renderings' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--reproject_mesh', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to reproject the mesh and render it in original image space, \
                            currently only available if rasterizer_type is standard, will add supports for pytorch3d \
                            after pytorch **stable version** supports non-squared images. \
                            default is False, means using the cropped image and its corresponding results')
    # texture options 
    parser.add_argument('--deca_path', default=None, type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA. \
                        please refer to https://github.com/YadiraF/DECA' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--lightTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to return lit albedo: that add estimated SH lighting to albedo')
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image, only do this when the face is near frontal and very clean!')
    # save
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--showParts', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to show head/hands crops in visualization' )
    parser.add_argument('--showWeight', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize the moderator weight on colored shape' )
    parser.add_argument('--saveGif', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize other views of the output, save as gif' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file' )
    parser.add_argument('--savePred', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save smplx prediction as pkl file' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
