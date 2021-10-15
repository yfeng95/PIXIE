import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.face_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg

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
    testdata = TestData(args.inputpath, iscrop=args.iscrop)

    #-- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)
    if args.deca_path:
        # if given deca code path, run deca to get face details
        sys.path.insert(0, args.deca_path)
        from decalib.deca import DECA
        deca = DECA(device=device)
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        util.move_dict_to_device(batch, device)
        batch['image'] = batch['image'].unsqueeze(0)
        name = batch['name']
        
        data = {
            'head': batch
        }
        param_dict = pixie.encode(data, keep_local=False, threthold=True)
        codedict = param_dict['head']
        opdict = pixie.decode(codedict, param_type='head')
        opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
        if args.saveObj or args.saveParam or args.savePred or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)

        # run deca if deca is available
        if args.deca_path is not None:
            _, deca_opdict, _ = deca.run(batch['imagepath'])
            flame_displacement_map = deca_opdict['displacement_map']
            opdict['displacement_map'] = visualizer.tex_flame2smplx(flame_displacement_map)
        if args.lightTex:
            visualizer.light_albedo(opdict)
        if args.extractTex:
            visualizer.extract_texture(opdict, data['head']['image'])
        if args.saveVis:
            visdict = visualizer.render_results(opdict, data['head']['image'], overlay=True)
            if args.showBody:
                body_vertices = pixie.decode_Tpose(codedict)
                body_vertices[:,:,1:] = -body_vertices[:,:,1:]
                cam = torch.zeros([1, 3], dtype=torch.float32, device=device)
                cam[:, 0] = 1.1; cam[:, 2] = -0.45
                trans_verts = util.batch_orth_proj(body_vertices, cam)
                visdict['body_shape'] = visualizer.render.render_shape(body_vertices, trans_verts)
            cv2.imwrite(os.path.join(savefolder, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=args.render_size))
        if args.saveObj:
            # save unposed body mesh
            if 'Tpose_vertices' not in opdict.keys():
                opdict['Tpose_vertices'] = pixie.decode_Tpose(codedict)
            visualizer.save_obj(os.path.join(savefolder, name, f'{name}.obj'), opdict, Tpose=True)
        if args.saveParam:
            util.save_pkl(os.path.join(savefolder, name, f'{name}_param.pkl'), codedict)
        if args.savePred:
            util.save_pkl(os.path.join(savefolder, name, f'{name}_prediction.pkl'), opdict) 
        if args.saveImages:
            for vis_name in visdict.keys():
                cv2.imwrite(os.path.join(savefolder, name, f'{name}_{vis_name}.jpg'), util.tensor2image(visdict[vis_name][0]))
                  
    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    parser.add_argument('-i', '--inputpath', default='TestSamples/face', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/face/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    # rendering option
    parser.add_argument('--render_size', default=224, type=int,
                        help='image size of renderings' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                    help='rasterizer type: pytorch3d or standard rasterizer' )
    # save
    parser.add_argument('--deca_path', default=None, type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA\
                        details of DECA: https://github.com/YadiraF/DECA' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--lightTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to return lit albedo: that add estimated SH lighting to albedo')
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image, only do this when the face is near frontal and very clean!')
    parser.add_argument('--showBody', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to show body mesh in visualization' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveGif', default=True, type=lambda x: x.lower() in ['true', '1'],
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
