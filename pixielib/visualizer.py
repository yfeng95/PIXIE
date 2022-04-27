import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.io import imread
import torchvision
import imageio
import pickle

from .models.FLAME import texture_flame2smplx
from .utils.renderer import set_rasterizer, SRenderY
from .utils import util
from .utils import rotation_converter
from .utils.config import cfg
from .utils.tensor_cropper import transform_points

import matplotlib.cm as cm
import matplotlib as matplotlib

# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# seismic
def color_map_color(value, cmap_name='cool', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(value)[:,:3]  # will return rgba, we take only first 3 so we get rgb
    # color = matplotlib.colors.rgb2hex(rgb)
    # rgb = np.tile(value[:,None], (1,3))
    # rgb = np.exp(rgb)
    return rgb

class Visualizer(object):
    ''' visualizer
    '''
    def __init__(self, render_size=1024, config=None, device='cuda:0', part='body', background=None, rasterizer_type='standard'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.render_size = render_size
        # cache data for smplx texture
        self.smplx_texture = imread(self.cfg.model.smplx_tex_path)/255.
        self.cached_data = np.load(self.cfg.model.flame2smplx_cached_path, allow_pickle=True, encoding = 'latin1').item()
        # Set up the renderer for visualization
        self.part = part
        self.rasterizer_type = rasterizer_type
        self._setup_renderer()
        # background
        if background is not None:
            self.bg = background
        else:
            self.bg = torch.ones([1,1,self.render_size, self.render_size], device=self.device, dtype=torch.float32)

    def _setup_renderer(self):
        ## setup raterizer
        set_rasterizer(self.rasterizer_type)
        uv_size = 1024
        if self.part == 'hand':
            topology_path = self.cfg.model.topology_smplx_hand_path
        else: # body or head
            topology_path = self.cfg.model.topology_smplxtex_path
            #  topology_path = self.cfg.model.topology_path
        self.render = SRenderY(self.render_size, obj_filename=topology_path, uv_size=uv_size, rasterizer_type=self.rasterizer_type).to(self.device)
        mask = imread(self.cfg.model.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.flame_face_eye_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)
        self.smplx_face_eye_mask = self.tex_flame2smplx(self.flame_face_eye_mask, np.zeros([uv_size, uv_size,1], dtype=np.float32))

        # face region mask in flame texture map
        mask = imread(self.cfg.model.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.flame_face_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)
        self.smplx_face_mask = self.tex_flame2smplx(self.flame_face_mask, np.zeros([uv_size,uv_size,1], dtype=np.float32))

        # head and hand ind for color rendering
        with open(self.cfg.model.mano_ids_path, 'rb') as f:
            hand_ind = pickle.load(f)
        head_ind = np.load(self.cfg.model.flame_ids_path)
        self.part_idx = {
            'head': head_ind,
            'left_hand': hand_ind['left_hand'],
            'right_hand': hand_ind['right_hand']
        }
        self.color_dict = {
            'head': [180/255., 0, 0],
            'left_hand': [0, 180/255., 0],
            'right_hand': [0, 0, 180/255.],
        }

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals=None):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        if coarse_normals is None:
            coarse_normals = util.vertex_normals(coarse_verts, self.render.faces.expand(coarse_verts.shape[0], -1, -1))
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z*self.smplx_face_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.smplx_face_mask + uv_coarse_normals*(1-self.smplx_face_mask)
        uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals

    def tex_flame2smplx(self, flame_texture, smplx_texture=None):
        ''' Convert flame texture to smplx texture
        '''
        if torch.is_tensor(flame_texture):
            device = flame_texture.device
            dtype = flame_texture.dtype
            flame_texture = flame_texture[0].cpu().detach().numpy().transpose(1,2,0)
            if smplx_texture is None:
                smplx_texture = self.smplx_texture
            smplx_texture = texture_flame2smplx(self.cached_data, flame_texture, smplx_texture)
            smplx_texture = torch.from_numpy(smplx_texture.transpose(2,0,1)).to(device).to(dtype).unsqueeze(0)
        else:
            smplx_texture = texture_flame2smplx(self.cached_data, flame_texture, self.smplx_texture)
        return smplx_texture

    def light_albedo(self, opdict):
        ''' add estimated SH lighting to face albedo
        '''
        # add estimated shading to albedo
        coarse_normals = util.vertex_normals(opdict['vertices'], self.render.faces.expand(opdict['vertices'].shape[0], -1, -1))
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
        uv_shading = self.render.add_SHlight(uv_coarse_normals, opdict['light'])
        uv_texture = opdict['albedo']*uv_shading
        opdict['texture'] = uv_texture

    def extract_texture(self, opdict, images):
        '''extract face texture from given cropped face image
        '''
        if 'texture' not in opdict.keys():
            self.light_albedo(opdict)
        uv_texture = opdict['texture']
        trans_verts = opdict['transformed_vertices']
        uv_pverts = self.render.world2uv(trans_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')
        ## TODO: poisson blending should give better-looking results
        # extract face only
        uv_texture_gt = uv_gt[:,:3,:,:]*self.smplx_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.smplx_face_eye_mask))
        # use this to extract whole texture from image
        # uv_texture_gt = uv_gt[:,:3,:,:]
        opdict['texture'] = uv_texture_gt

    def recover_position(self, opdict, batch, tform, original_image):
        ''' transofrm mesh back to original image space
        '''
        points_scale = batch['image'].shape[2:]
        trans_verts = opdict['transformed_vertices']
        trans_verts = transform_points(trans_verts, tform, points_scale)
        _, _, h, w = original_image.shape
        trans_verts[:,:,0] = trans_verts[:,:,0]/w*2 - 1
        trans_verts[:,:,1] = trans_verts[:,:,1]/h*2 - 1
        opdict['transformed_vertices'] = trans_verts
        batch['image_hd'] = original_image

    def render_results(self, opdict, input_images, overlay=False, light_on=False, moderator_weight=None, use_deca=False, background=None):
        ''' rendering pixie predictions
        Args:
            opdict: output from pixie that contains smplx vertices
            inpit_images: test image, for rending background
        '''
        batch_size, nv, _ = opdict['vertices'].shape
        if overlay:
            bg = input_images
        elif background is not None:
            bg = background
        else:
            bg = self.bg

        # render geometry
        _, _, h, w = input_images.shape
        shape_images, normal_images, grid = self.render.render_shape(opdict['vertices'], opdict['transformed_vertices'], background=bg, return_grid=True, h=h, w=w)
        visdict={
            'inputs': input_images,
            'shape_images': shape_images
        }

        # if detail displacement map is available after running deca
        if 'displacement_map' in opdict.keys():
            uv_detail_normals = self.displacement2normal(opdict['displacement_map'], opdict['vertices'])
            face_detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)
            face_mask = F.grid_sample(self.smplx_face_mask, grid, align_corners=False)
            normal_images = face_detail_normal_images*face_mask + normal_images*(1-face_mask)
            shape_detail_images = self.render.render_shape(opdict['vertices'], opdict['transformed_vertices'], detail_normal_images=normal_images, background=bg, h=h, w=w)
            visdict['shape_detail_image'] = shape_detail_images
            opdict['normal_map'] = uv_detail_normals.detach()
        elif use_deca:
            visdict['shape_detail_image'] = shape_images

        if moderator_weight is not None:
            #  colors = torch.tensor([180/255., 180/255., 180/255.])[None, None, :].repeat(batch_size, nv, 1).float().to(self.device)
            colors = torch.tensor([1.0, 1.0, 0.9])[None, None, :].repeat(batch_size, nv, 1).float().to(self.device)
            for part in moderator_weight.keys():
                # import ipdb; ipdb.set_trace()
                weight = moderator_weight[part].cpu().numpy()[:,0]
                curr_color = torch.from_numpy(color_map_color(weight)).float().to(self.device)
                colors[:, self.part_idx[part], :] = curr_color[:,None,:]
            opdict['vertex_colors'] = colors
            face_colors = util.face_vertices(colors, self.render.faces.expand(batch_size, -1, -1))
            color_shape_images = self.render.render_shape(opdict['vertices'], opdict['transformed_vertices'], background=bg, colors = face_colors, h=h, w=w)
            visdict['color_shape_images'] = color_shape_images

        # import ipdb; ipdb.set_trace()
        uvmap = opdict['texture'] if 'texture' in opdict.keys() else opdict['albedo']
        # TODO: use which lighting to render color image?
        # currently none, maybe use estimated lighting (SH) or set directional lighting?
        lights = opdict['light'] if light_on else None
        render_ops = self.render(opdict['vertices'], opdict['transformed_vertices'], uvmap, background=bg, lights=lights, h=h, w=w)
        rendered_images = render_ops['images']
        # visdict['rendered_images'] = rendered_images
        return visdict

    def rotate_results(self, opdict, result_type='color', light_on=False, visdict=None, savepath='vis.gif'):
        ''' view body mesh in different poses, save in gif
        Args:
            opdict: output from pixie that contains smplx vertices
            result_type: color or shape, rendered image type
        '''
        if visdict is None:
            visdict = {}
        # rotate mesh and rendering
        writer = imageio.get_writer(savepath, mode='I')
        for yaw_angle in range(0, 361, 30):
            euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
            euler_pose[:,1] = yaw_angle
            global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
            vertices = torch.matmul(opdict['vertices'], global_pose.transpose(1,2))
            transformed_vertices = util.batch_orth_proj(vertices, opdict['cam'])
            if result_type=='color':
                uvmap = opdict['texture'] if 'texture' in opdict.keys() else opdict['albedo']
                lights = opdict['lights'] if light_on else None
                render_ops = self.render(vertices, transformed_vertices, uvmap, background=self.bg, lights=lights)
                visdict['other_view'] = render_ops['images']
            else:
                shape_images = self.render.render_shape(vertices, transformed_vertices, background=self.bg)
                visdict['other_view'] = shape_images
            grid_image = self.visualize_grid(visdict, size=self.render_size)
            # grid_image = self.visualize_grid(visdict, size=512)
            writer.append_data(grid_image[:,:,[2,1,0]])

    def save_obj(self, objpath, opdict, k=0, Tpose=False):
        faces = self.render.faces.squeeze().cpu().numpy()
        uvcoords = self.render.raw_uvcoords.squeeze().cpu().numpy()
        uvfaces = self.render.uvfaces.squeeze().cpu().numpy()
        if Tpose:
            vertices = opdict['Tpose_vertices'][k].cpu().numpy()
        else:
            vertices = opdict['vertices'][k].cpu().numpy()
            #  vertices = opdict['out_vertices'][k].cpu().numpy()

        uvmap = opdict['texture'] if 'texture' in opdict.keys() else opdict['albedo']
        uvmap = util.tensor2image(uvmap[k])
        if 'vertex_colors' in opdict.keys():
            colors = opdict['vertex_colors'].squeeze().cpu().numpy()
        else:
            colors = None    
        util.write_obj(
            objpath, vertices, faces,
            colors=colors,
            texture=uvmap,
            uvcoords=uvcoords,
            uvfaces=uvfaces,
            inverse_face_order=False,
            normal_map=opdict.get('normal_map'),
        )

    def visualize_grid(self, visdict, savepath=None, size=224, dim=2, return_gird=True):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        grids = {}
        for key in visdict:
            _,_,h,w = visdict[key].shape
            if dim == 2:
                new_h = size; new_w = int(w*size/h)
            elif dim == 1:
                new_h = int(h*size/w); new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath:
            cv2.imwrite(savepath, grid_image)
        if return_gird:
            return grid_image

