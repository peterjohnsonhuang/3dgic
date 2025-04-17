# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torchvision.transforms as Trans
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from gaussian_renderer import render
# from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import GaussianModel
import numpy as np
from PIL import Image
import cv2
from utils.loss_utils import masked_l1_loss, l1_loss, ssim, loss_cls_3d
from random import randint
import lpips
import json
from sklearn.decomposition import PCA
import colorsys
from scene.gamma_trans import LearningGammaTransform
from scene.derect_light_sh import DirectLightEnv

from edit_object_removal import points_inside_convex_hull
from utils.ggutils import visualize_obj, feature_to_rgb



def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        # formatted_points.append("%f %f %f 0\n" % (point[1], point[0], point[2]))
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3]*255, point[4]*255, point[5]*255))

    out_file = open(ply_filename, "w")
    # out_file.write('''ply
    # format ascii 1.0
    # element vertex %d
    # property float x
    # property float y
    # property float z
    # property uchar alpha
    # end_header
    # %s
    # ''' % (len(points), "".join(formatted_points)))
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

def depth_image_to_point_cloud(source_depths, source_intrinsics, source_c2ws, img_wh, source_image, inpaint_mask2d):
    ''' 
    source_depth: [N, H, W]
    source_intrinsics: [N, 3, 3]
    source_c2ws: [N, 4, 4]
    target_intrinsics: [3, 3]
    target_w2c: [4, 4]
    img_wh: [2]
    grid_size: int
    return: depth map [H, W]
    
    '''

    source_depths = source_depths.unsqueeze(0).float()
    source_intrinsics = source_intrinsics.unsqueeze(0).float()
    source_c2ws = source_c2ws.unsqueeze(0).float()
    H,W = img_wh
    N = source_depths.shape[0]
    # print(N)
    points = []
    # print(source_depths.shape, source_image.shape)

    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing="ij"
    )  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1).to(source_intrinsics.device), xs.reshape(-1).to(source_intrinsics.device)

    for num in range(N):
        # Might need to change this to be more general (too small or too big value are not good)
        mask = source_depths[num] > 0

        dirs = torch.stack(
        [
            (xs - source_intrinsics[num][0, 2]) / source_intrinsics[num][0, 0],
            (ys - source_intrinsics[num][1, 2]) / source_intrinsics[num][1, 1],
            torch.ones_like(xs),
        ],
        -1,
        )
        # print(dirs.shape, source_c2ws.shape)
        rays_dir = (
            dirs @ source_c2ws[num][:3, :3].t()
        )
        
        rays_orig = source_c2ws[num][:3, -1].clone().reshape(1, 3).expand(rays_dir.shape[0], -1)
        rays_orig = rays_orig.reshape(H,W,-1)[mask]
        rays_depth = source_depths[num].reshape(H,W,-1)[mask]
        rays_dir = rays_dir.reshape(H,W,-1)[mask]
        ray_pts = rays_orig + rays_depth * rays_dir
        points.append(ray_pts.reshape(-1,3))
        rgb = source_image[mask].reshape(-1,3)
        # print(mask.shape)
        inpaint_mask2d = inpaint_mask2d[mask].reshape(-1).nonzero()

        del rays_orig, rays_depth, rays_dir, ray_pts, dirs, mask

    points = torch.cat(points,0).reshape(-1,3)
    
    return torch.cat((points.squeeze(0),rgb),1)[inpaint_mask2d].squeeze(1)




def world_to_new_view(points, target_intrinsics, target_w2cs, return_depth = False):
    ''' 
    source_depth: [N, H, W]
    source_intrinsics: [N, 3, 3]
    source_c2ws: [N, 4, 4]
    target_intrinsics: [3, 3]
    target_w2c: [4, 4]
    img_wh: [2]
    grid_size: int
    return: depth map [H, W]
    
    '''

    
    point_xyz = torch.ones((points.shape[0],4)).to(target_intrinsics.device)
    point_xyz[:,:3] = points[:,:3].clone().detach()
    point_xyz = point_xyz.T
    xyz_c = target_w2cs@point_xyz
    lambda_uv = target_intrinsics@xyz_c[:3]
    uv = torch.ones((points.shape[0],2)).T.to(target_intrinsics.device)
    uv[0] = lambda_uv[0]/lambda_uv[2]
    uv[1] = lambda_uv[1]/lambda_uv[2]
    uv = uv.long().float()
    # print(uv)
    # input()
    # print(point_xyz)
    # input()
    # print(target_w2cs)
    # input()

    if return_depth == True:
        return torch.cat((uv.T, points[:,3:]),1), lambda_uv[2]
    else:
        return torch.cat((uv.T, points[:,3:]),1)

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def get_init_points(cam, default_depth=False, custom_mask = None):
            
    print(cam.image_name)
    
    
    depth = (cam.depths).float().permute(2,0,1)

    # gt_obj = depth[-1].long().cuda()
    # gt_mask = (gt_obj == 78)

    
    # print(depth.shape)
    # blurrer = Trans.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1))
    # depth = blurrer(depth)
    depth = depth[0]
    # print(depth.shape)
    if default_depth == True:
        depth  = torch.ones_like(depth)
    # print(depth)
    # depth = (viewpoint_cam.depths).float()
    # print(viewpoint_cam.objects.max())
    # input()

    if custom_mask is not None:
        mask2d = custom_mask
    else:    
        mask2d = (cam.objects >128).to(depth.device).squeeze()

    # mask2d = gt_mask
    # print(mask2d)
    # print(mask2d.shape)
    # input()
    # print(viewpoint_cam.objects)
    # depth_point = depth[mask2d>0]
    # mask_index = mask2d.nonzero(as_tuple=True)
    # print(mask_index)
    # mask_index = mask2d.reshape(-1).nonzero()
    # print(len(mask_index[0]), depth[mask_index].shape)
    # mask_depth = depth[mask_index]
    gt_image = cam.original_image.cuda()
    # print(gt_image.shape)
    gt_image = gt_image.permute((1,2,0))
    # print(depth, viewpoint_cam.objects.shape)
    # print(depth.max())
    R,T,FovY,FovX, width, height = cam.R, cam.T, cam.FoVy, cam.FoVx, cam.image_width, cam.image_height
    f_x = fov2focal(FovX, width)
    f_y = fov2focal(FovY, height)
    # print(f_x, f_y)
    # camera_center = viewpoint_cam.camera_center
    # print(camera_center)
    # print(T)
    c_y = height/2.0
    c_x = width/2.0
    # P = np.zeros((4, 4))
    # P[0:3,0:3] = R
    # P[3,0:3] = T
    # P[3,3] = 1
    A = torch.Tensor([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,0,1.0]]).to(depth.device)
    # print(mask_index[0].shape)
    # extrinsic = cam.extrinsics
    extrinsic = cam.camera_back_project
    # print(extrinsic)
    # exit()
    # mask_index = torch.cat([mask_index[0].unsqueeze(1), mask_index[1].unsqueeze(1), torch.ones((len(mask_index[0]),1), device = mask_index[0].device)],1)
    # print(mask_index.shape)
    # mask_depth = mask_depth.unsqueeze(1).repeat(1,3)
    
    # points = depth_image_to_point_cloud(width, height, depth.cpu().numpy(), 1, A, P, mask2d.cpu().numpy())
    # points = depth_image_to_point_cloud(width, height, mask_depth.cpu().numpy(), 1, A, R,T, mask_index.cpu().numpy(), gt_image.cpu().numpy())
    points = depth_image_to_point_cloud(depth, A, extrinsic, depth.shape, gt_image, mask2d)
    # print(points.shape)
    # input()
    points_all = depth_image_to_point_cloud(depth, A, extrinsic, depth.shape, gt_image, torch.ones_like(mask2d))
    # points_all = points
    # print(points.shape)
    # input()
    # points = points[mask_index].squeeze(1)
    
    
    del depth, gt_image
    return points, points_all

def finetune_inpaint(is_pbr, pbr_kwargs, dataset, pipe, opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, finetune_iteration,exp_setting):
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # get 3d gaussians idx corresponding to select obj id
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        mask3d = mask3d.float()[:,None,None]

    # fix some gaussians
    points = []
    points_all = None
    ref_view_name = []
    # viewpoint_stack = views.copy()
    # print(len(viewpoint_stack))
    # exit()
    
    # if _use_ref == False:
    #     for view_id in range(100):
    #         ref_view_name.append(viewpoint_stack[view_id].image_name)

    for ref_id in _ref_id:
        viewpoint_stack = views.copy()


        cam = viewpoint_stack.pop(ref_id-1)
        depth = (cam.depths).float().permute(2,0,1)
        gt_obj = depth[-1].long().cuda()
        gt_mask = (gt_obj == selected_obj_ids[0])
        ref_view_name.append(cam.image_name)
        # point, _ = get_init_points(cam, custom_mask=gt_mask)
        point, _ = get_init_points(cam)
        points.append(point)


    # cam = viewpoint_stack.pop(61)
    # ref_view_name.append(cam.image_name)
    # depth = (cam.depths).float().permute(2,0,1)
    # gt_obj = depth[-1].long().cuda()
    # gt_mask = (gt_obj == selected_obj_ids[0])
    # point2, _ = get_init_points(cam, custom_mask=gt_mask)


    # cam = viewpoint_stack.pop(38)
    # ref_view_name.append(cam.image_name)
    # depth = (cam.depths).float().permute(2,0,1)
    # gt_obj = depth[-1].long().cuda()
    # gt_mask = (gt_obj == selected_obj_ids[0])
    # point3, _ = get_init_points(cam, custom_mask=gt_mask)

    # cam = viewpoint_stack.pop(57)
    # ref_view_name.append(cam.image_name)
    # depth = (cam.depths).float().permute(2,0,1)
    # gt_obj = depth[-1].long().cuda()
    # gt_mask = (gt_obj == selected_obj_ids[0])
    # point4, _ = get_init_points(cam, custom_mask=gt_mask)

    # cam = viewpoint_stack.pop(76)
    # ref_view_name.append(cam.image_name)
    # depth = (cam.depths).float().permute(2,0,1)
    # gt_obj = depth[-1].long().cuda()
    # gt_mask = (gt_obj == selected_obj_ids[0])
    # point5, _ = get_init_points(cam, custom_mask=gt_mask)
    if len(points) > 1:
        points = torch.cat(points)
    elif len(points) == 1:
        points = points[0]
    else:
        print("no any ref points")
    # points_all = torch.cat((all1, all2))
    # points = torch.cat((point2, point4, point5))
    # points_all = all1
    # points = point1
    # print(points)
    # input()
    # print(points.shape)
    # print(gaussians._xyz)

    # camera_center = camera_center.repeat(points.shape[0],1)
    # points[:,:3]+=camera_center
    




    # write_point_cloud('./test_point_1.ply', all1)
    # write_point_cloud('./test_point_2.ply', point2)
    # write_point_cloud('./test_point_3.ply', point3)
    # write_point_cloud('./test_point_4.ply', point4)
    # write_point_cloud('./test_point_5.ply', point5)
    write_point_cloud('./test_point_1.ply', points)
    # exit()
    # print()

    # W2C = getWorld2View2(cam.R, cam.T)
    # C2W = np.linalg.inv(W2C)
    # center = C2W[:3, 3:4]
    # print(camera_center)
    # print(f_x, f_y)
    # print(opt.skip_init)
    # exit()
    # print(opt.skip_init == False)
    # print(opt.skip_init)
    # print(point.size())
    if opt.skip_init == "False":
        choice_id =  torch.randperm(points.size(0))[:int(points.size(0))]
        points_selected = points[choice_id]
        gaussians.inpaint_setup(opt,mask3d, points_selected[:,:3], is_pbr=is_pbr)
        # gaussians.inpaint_setup(opt,mask3d, None, is_pbr=is_pbr)
    else:
        print("continue inpainting")
        gaussians.training_setup(opt)
    
    # gaussians.training_setup(opt)


    # if is_pbr:
    # #     gaussians.inpaint_setup(opt,mask3d, points_selected[:,:3], is_pbr=True)
    #     gaussians.training_setup(opt)
        
    # else:
        
        

    #     # points = points_selected ### if OOM
        
    #     gaussians.inpaint_setup(opt,mask3d, points_selected[:,:3], is_pbr=True)
    #     # gaussians.inpaint_setup(opt,mask3d)
    #     # print(gaussians._xyz.shape)
    #     # input()
    #     # input()
    #     # gaussians.inpaint_setup(opt,mask3d)
    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()

    
    """
    Setup PBR components
    """
    if is_pbr:
        render_fn = render_fn_dict["neilf"]
    else:
        render_fn = render_fn_dict["render"]
    # print(gaussians._xyz.shape)

    if _use_ref == True:
        print("using reference view:", ref_view_name)
    print(iterations)
    for iteration in range(iterations):

        # print((gaussians._xyz[0].requires_grad == True))
        # exit()
        viewpoint_stack = views.copy()

        
             

        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        depth = (viewpoint_cam.depths).float().permute(2,0,1)
        cam_name = viewpoint_cam.image_name
        gt_obj = depth[-1].long().cuda()
        gt_mask = (gt_obj == selected_obj_ids[0])
        mask2d = (viewpoint_cam.objects > 128).squeeze().cuda()
        if torch.equal(mask2d.long(), torch.zeros_like(mask2d)):
            if depth.shape[0] > 3:
                
                mask2d = gt_mask
            else:
                mask2d[0,0] = True
                print("haha")

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        

        
        # gt_obj = depth[-1].long().cuda()
        # gt_mask = (gt_obj == selected_obj_ids[0])
        # mask2d = gt_mask


        if not torch.equal(mask2d.long(), torch.zeros_like(mask2d)):

            if is_pbr:
                # gt_obj = depth[-1].long().cuda()
                # gt_mask = (gt_obj == selected_obj_ids[0])
                render_pkg = render_fn(viewpoint_cam, gaussians, pipeline, background,
                                opt=opt, is_training=True, dict_params=pbr_kwargs, mask=mask2d)
            else:
                render_pkg = render_fn(viewpoint_cam, gaussians, pipeline, background)
            image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

            if is_pbr and _use_base:
                image = render_pkg["base_color"]
            
            pred_depth = render_pkg["depth"]
            # print(pred_depth)
            # print(depth[:3])
            # exit()

            R,T,FovY,FovX, width, height = viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.FoVy, viewpoint_cam.FoVx, viewpoint_cam.image_width, viewpoint_cam.image_height
            # print(FovX, FovY)
            # input()
            f_x = fov2focal(FovX, width)
            f_y = fov2focal(FovY, height)
            # print(f_x, f_y)
            # camera_center = viewpoint_cam.camera_center
            # print(camera_center)
            # print(T)
            c_y = height/2.0
            c_x = width/2.0
            P = np.zeros((4, 4))
            P[0:3,0:3] = R
            P[3,0:3] = T
            P[3,3] = 1
            A = torch.Tensor([[f_x, 0, c_x],
                            [0, f_y, c_y],
                            [0,0,1.0]]).to(image.device)
            # print(mask_index[0].shape)
            extrinsic = viewpoint_cam.camera_back_project
            wtc_project = viewpoint_cam.world_view_transform.T
            # print(T)
            if points_all != None:
                target_point_ref, depth_ref = world_to_new_view(points_all, A, torch.Tensor(wtc_project).to(image.device), return_depth=True)
            else:
                target_point_ref, depth_ref = world_to_new_view(points, A, torch.Tensor(wtc_project).to(image.device), return_depth=True)

            # target_point_ref = world_to_new_view(points, A, torch.Tensor(P.T).to(image.device))
            # print(target_point_ref.shape)
            # input()
            # print(wtc_project.shape)
            # print(depth_ref)
            # input()
            gt_image = viewpoint_cam.original_image.cuda()
            gt_swap_temp = gt_image.clone().detach().permute(1,2,0)
            gt_swap = gt_image.clone().detach().permute(1,2,0)
            # print(gt_swap.shape)
            # input()
            # target_point_ref = target_point_ref[target_point_ref[:,1]<728 & target_point_ref[:,0]<984]
            # b1, b2, b3, b4 = (target_point_ref[:,0]<=984), (target_point_ref[:,0]>=0), (target_point_ref[:,1]<=728), (target_point_ref[:,1]>=0)
            # print(gt_image.shape)
            b1, b2, b3, b4 = (target_point_ref[:,0]<=(gt_image.shape[2]-1)), (target_point_ref[:,0]>=0), (target_point_ref[:,1]<=(gt_image.shape[1]-1)), (target_point_ref[:,1]>=0)
            selected = (b1*b2*b3*b4).nonzero(as_tuple=True)[0]
            target_point_ref_clone = target_point_ref.detach().clone()[selected]
            depth_ref_clone = depth_ref.detach().clone()[selected].unsqueeze(-1).repeat(1,3)
            depth_swap = (torch.ones_like(depth[:3])*1000000.0).permute(1,2,0)
            swap_index = target_point_ref_clone.clone().long()[:,:2].T



            # gt_swap_temp[(swap_index[1], swap_index[0])] = target_point_ref_clone[:,2:]
            # depth_swap[(swap_index[1], swap_index[0])] = depth_ref_clone
            # depth_mask = torch.ge(depth[:3].permute(1,2,0), depth_swap).int()
            # updated_swap_index = depth_mask[:,:,0].nonzero(as_tuple=True)
            


            if _use_ref == True:
                gt_swap[(swap_index[1], swap_index[0])] = target_point_ref_clone[:,2:]
                # gt_swap[(updated_swap_index[0], updated_swap_index[1])] = gt_swap_temp[(updated_swap_index[0], updated_swap_index[1])]
            
            # else:
            #     print("not using ref")


            
            gt_swap = gt_swap.permute(2,0,1)
            # print(gt_swap.shape)
            # input()

            # swap_index = target_point_ref.clone().detach().long()[:,:2].T
            # # print(swap_index.shape)
            # # print(swap_index[0].max(), swap_index[1].max())
            # gt_swap[(swap_index[1].clamp(0,728), swap_index[0].clamp(0,984))] = target_point_ref[:,2:]
            # # input()
            # # gt_image[]
            # gt_swap = gt_swap.permute(2,0,1)


            
            
            # print(image.shape, gt_image.shape, mask2d.shape)
            # print(image.device, gt_image.device, mask2d.device)
            Ll1 = masked_l1_loss(image, gt_image, ~mask2d)
            
            # del gt_image

            # print('haha')
            # print(mask2d.shape)
            # print(mask2d)

            bbox = mask_to_bbox(mask2d)
            # print(bbox)
            cropped_image = crop_using_bbox(image, bbox)
            cropped_gt_image = crop_using_bbox(gt_swap, bbox)

            if min(cropped_image.size()) <= 64:
                pd64 = (64,64,64,64)
                cropped_gt_image = torch.nn.functional.pad(cropped_gt_image, pd64)
                cropped_image = torch.nn.functional.pad(cropped_image, pd64)
            
            # blurrer = Trans.GaussianBlur(kernel_size=(9, 9), sigma=(5, 5))
            # blurrer = Trans.GaussianBlur(kernel_size=(3, 3), sigma=(1, 1))
            # cropped_gt_image = blurrer(cropped_gt_image)
            # cropped_image = blurrer(cropped_image)
            

            # print(cropped_image.shape)

            # torchvision.utils.save_image((gt_swap).detach().cpu(), 'test_{}.png'.format(iteration))
            K = 2
            rendering_patches = divide_into_patches(cropped_image[None, ...], K)
            gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
            # print(gt_patches.shape)
            lpips_loss = LPIPS(rendering_patches.squeeze()*2-1,gt_patches.squeeze()*2-1).mean()


            ### semantic loss (calculate only if original object mask and gt image are available)
            loss_obj_3d = 0
            loss_obj = 0
            # print(opt.reg3d_interval)
            if depth.shape[0] > 3:
                gt_obj = depth[-1].long().cuda()
                # gt_obj = (gt_obj * (~(gt_obj == selected_obj_ids[0])))
                logits = classifier(objects)
                # print(logits.shape)
                # print(cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).shape)
                # input()
                loss_obj = (cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze()*(~(gt_obj == selected_obj_ids[0]))).mean()
                loss_obj = loss_obj / torch.log(torch.tensor(logits.shape[0]))

                ### reg_loss_3d
                loss_obj_3d = 0
                if iteration % opt.reg3d_interval == 0:
                    # regularize at certain intervals
                    logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
                    prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                    loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
                    # print('sssss')
                else:
                    loss_obj_3d = 0
                # print(loss_obj)
            else:
                loss_obj = 0

            # print(depth.shape)
            # input()

            if (cam_name in ref_view_name) and (depth.shape[0] > 3):
                # print(image.shape)
                if is_pbr:
                    loss_2d = masked_l1_loss(render_pkg["pbr"],depth[3:6], mask2d)
                    # loss_2d = l1_loss(depth[3:6], render_pkg["pbr"])
                    loss_2d += masked_l1_loss(render_pkg["render"],depth[3:6], mask2d)
                else:
                    loss_2d = l1_loss(image,depth[3:6])
                
                
                print(loss_2d)
                # loss_2d += loss_depth * 0.1 
                print(loss_depth)
                
                if _use_ref == True:
                    loss_2d = loss_2d*5.0
                torchvision.utils.save_image((depth[3:6]).detach().cpu(), 'test_gt.png')
                
            else:
                loss_2d = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss

            pred_depth = render_pkg["depth"]
            loss_depth = l1_loss(pred_depth, depth[:3]) * 0.1
            loss = loss_2d*2.0 + loss_obj*1.0 + loss_obj_3d + loss_depth
            # print(loss)
            # adding neilf loss
            if is_pbr:
                loss += render_pkg["loss"]
            # print(loss)
            loss.backward()
            with torch.no_grad():
                if (iteration <= 51) and (iteration != 0):
                    # Keep track of max radii in image-space for pruning
                    # print(gaussians.max_radii2D.device)
                    # print(visibility_filter.device)
                    # print(radii.device)
                    # print()
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # if  iteration % 300 == 0:
                    
                    if  iteration % 1000 == 0:
                        size_threshold = 20 
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.000, cameras_extent, size_threshold,  opt.densify_grad_normal_threshold)
                        print(gaussians._xyz.shape)
                    # if (iteration >= 1000) and (iteration % 600) == 0:
                    #     gaussians.prune(0.005, cameras_extent, size_threshold)
                    #     print(gaussians._xyz.shape)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

        else:
            if is_pbr:
                render_pkg = render_fn(viewpoint_cam, gaussians, pipeline, background,
                                opt=opt, is_training=True, dict_params=pbr_kwargs, mask=mask2d)
            else:
                render_pkg = render_fn(viewpoint_cam, gaussians, pipeline, background)
            image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

            
            # Object Loss
            gt_obj = viewpoint_cam.objects.cuda().long()
            if depth.shape[0] > 3:
                gt_obj = depth[-1].long().cuda()
            
            logits = classifier(objects)
            loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            loss_obj = loss_obj / torch.log(torch.tensor(logits.shape[0]))  # normalize to (0,1)

            loss_obj_3d = 0

            if (cam_name in ref_view_name) and (depth.shape[0] > 3):
                # print(image.shape)
                if is_pbr:
                    loss_2d = l1_loss(depth[3:6], render_pkg["pbr"])
                    loss_2d += render_pkg["loss"]
                else:
                    loss_2d = l1_loss(depth[3:6], image)
            else:
                loss_2d = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss
            if iteration % opt.reg3d_interval == 0:
                # regularize at certain intervals
                logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
                prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
                loss = loss_2d + (loss_obj*0.5 + loss_obj_3d)
            else:
                loss = loss_2d + loss_obj*0.5

            # Loss
            # tb_dict = render_pkg["tb_dict"]
            # loss += render_pkg["loss"]
            try:
                loss.backward(retain_graph = True)
                with torch.no_grad():
                    if (iteration <= 2000) and (iteration != 0):
                        # Keep track of max radii in image-space for pruning
                        # print(gaussians.max_radii2D.device)
                        # print(visibility_filter.device)
                        # print(radii.device)
                        # print()
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        # if  iteration % 300 == 0:
                        
                        if  iteration % 5000 == 0:
                            size_threshold = 20 
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold,  opt.densify_grad_normal_threshold)
                            print(gaussians._xyz.shape)
                        # if (iteration >= 1000) and (iteration % 300) == 0:
                        #     gaussians.prune(0.005, cameras_extent, size_threshold)
                        #     print(gaussians._xyz.shape)
                        
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            except:
                # print(cam_name)
                loss = 0

        torch.cuda.empty_cache()
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)

    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint/{}_iteration_{}".format(exp_setting, iteration))
    # point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    torch.save((gaussians.capture(), iteration),
                os.path.join(point_cloud_path, "chkpnt" + str(iteration) + ".pth"))
    for com_name, component in pbr_kwargs.items():
        print(pbr_kwargs)
        try:
            torch.save((component.capture(), iteration),
                        os.path.join(point_cloud_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
        except:
            pass
    return gaussians




def render_set(is_pbr, pbr_kwargs, model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    if is_pbr:
        render = render_fn_dict["neilf"]
    else:
        render = render_fn_dict["render"]
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours{}".format(iteration), "depth")
    depth_real_path = os.path.join(model_path, name, "ours{}".format(iteration), "depth_removal")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    base_color_path = os.path.join(model_path, name, "ours{}".format(iteration), "base_color")
    pbr_path = os.path.join(model_path, name, "ours{}".format(iteration), "pbr")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_real_path, exist_ok=True)
    makedirs(base_color_path, exist_ok=True)
    makedirs(pbr_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if is_pbr:
            results = render(view, gaussians, pipeline, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
        else:
            results = render(view, gaussians, pipeline, background)
        # results = render(view, gaussians, pipeline, background, depth=True)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        if is_pbr:
            base_color = results["base_color"]
            pbr = results["pbr"]

        # print(results["name"])
        image_name = results["name"]
        # input()
        if "frame" in image_name:
            idx = int(image_name[6:])-1
        else:
            idx = int(image_name)


        # removed_rendering_obj = removed_results["render_object"]
        depth_real = results["depth"]
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        # logits = classifier(removed_rendering_obj.cuda())
        pred_obj = torch.argmax(logits,dim=0).cpu()
        # pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        if view.objects is None:
            gt_objects = torch.zeros_like(depth_real).squeeze(0)
        else:
            gt_objects = view.objects.squeeze(0)
        # remove_obj = ((gt_objects * (pred_obj == select_obj_id[0])) > 0).int().cpu()*255.0
        # remain_obj = (gt_objects * (~(gt_objects == select_obj_id[0]))).cpu().int().numpy().astype(np.uint8)
        remain_obj = (gt_objects).cpu().int().numpy().astype(np.uint8)
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        # print(rendered_depth.shape)
        # print(depth_real.max(), depth_real.min())
        # input()

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        # Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        # Image.fromarray(remain_obj).save(os.path.join(remain_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        gt_shape = gt.shape

        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(remove_obj, os.path.join(remove_obj_path, '{0:05d}'.format(idx) + ".png"))
        if is_pbr:
            torchvision.utils.save_image(base_color, os.path.join(base_color_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(pbr, os.path.join(pbr_path, '{0:05d}'.format(idx) + ".png"))
        idx+=1
        torchvision.utils.save_image(rendering, os.path.join(render_path, 'frame_{0:05d}'.format(idx) + ".png"))
        if is_pbr:
            torchvision.utils.save_image(base_color, os.path.join(base_color_path, 'frame_{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(pbr, os.path.join(pbr_path, 'frame_{0:05d}'.format(idx) + ".png"))
        

        d_min, d_max = depth_real.min(), depth_real.max()

        # print(d_min, d_max)
        # input()
        depth_real = (depth_real.float().clamp(0,d_max) - d_min)/(d_max-d_min)
        torchvision.utils.save_image(depth_real, os.path.join(depth_real_path, '{0:05d}'.format(idx-1) + ".png"))
        torch.save((d_min,d_max),os.path.join(depth_real_path, '{0:05d}_range'.format(idx-1) + ".pt"))

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    size = (gt_shape[-1]*5,gt_shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        idx = int(file_name.split('.')[0])+1
        gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,'frame_'+'{0:05d}.png'.format(idx))))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:,:,::-1])

    writer.release()


def inpaint(is_pbr : bool, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int, exp_setting: str):
    # 1. load gaussian checkpoint
    if is_pbr:
        gaussians = GaussianModel(dataset.sh_degree, render_type="neilf")
        if args.checkpoint:
            # print(args.checkpoint)
            # input()
            print("Create Gaussians from checkpoint {}".format(args.checkpoint))
            first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        scene = Scene(dataset, gaussians, shuffle=False)
    else:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    # TODO: load classifier checkpoint need to be searched from last
    from utils.system_utils import searchForMaxIteration
    loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    # print(loaded_iter)
    # input()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_{}".format(loaded_iter),"classifier.pth")))
    for param in classifier.parameters():
        param.requires_grad = False
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pbr_kwargs = dict()

    if is_pbr:
        pbr_kwargs['sample_num'] = pipeline.sample_num
        if dataset.use_global_shs == 1:
            print("Using global incident light for regularization.")
            direct_env_light = DirectLightEnv(dataset.global_shs_degree)

            if args.checkpoint:
                env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
                print("Trying to load global incident light from ", env_checkpoint)
                if os.path.exists(env_checkpoint):
                    direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")

            direct_env_light.training_setup(opt)
            pbr_kwargs["env_light"] = direct_env_light

        if opt.use_ldr_image:
            print("Using learning gamma transform.")
            gamma_transform = LearningGammaTransform(opt.use_ldr_image)

            if args.checkpoint:
                gamma_checkpoint = os.path.dirname(args.checkpoint) + "/gamma_" + os.path.basename(args.checkpoint)
                print("Trying to load gamma checkpoint from ", gamma_checkpoint)
                if os.path.exists(gamma_checkpoint):
                    gamma_transform.create_from_ckpt(gamma_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")

            gamma_transform.training_setup(opt)
            pbr_kwargs["gamma"] = gamma_transform

        if opt.finetune_visibility:
            gaussians.finetune_visibility()

    # 2. inpaint selected object
    gaussians = finetune_inpaint(is_pbr, pbr_kwargs, dataset, pipeline, opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh, finetune_iteration,exp_setting)

    # 3. render new result
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    # scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/iteration_'+str(finetune_iteration-1), shuffle=False)
    del scene
    with torch.no_grad():
        torch.cuda.empty_cache()
        scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/{}_iteration_'.format(exp_setting)+str(finetune_iteration-1), shuffle=False)
        if not skip_train:
             render_set(is_pbr, pbr_kwargs, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test:
             render_set(is_pbr, pbr_kwargs, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)


    args = get_combined_args(parser)
    print("Rendering " + args.model_path, "with" + args.type + "rendering")

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10_000)
    args.source_path = config.get("source_path", "./data/bear")
    args.exp_setting = config.get("exp_setting", "depth_init")
    # print(config.get("skip_init", "False"))
    # input()
    args.skip_init = config.get("skip_init", "False")
    try:
        args.reg3d_interval = config.get("reg3d_interval", 2)
        args.reg3d_k = config.get("reg3d_k", 5)
        args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
        args.reg3d_max_points = config.get("reg3d_max_points", 300000)
        args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    except:
        pass

    _use_ref = bool(config.get("use_ref", 1))
    _use_base = bool(config.get("use_base", 1))
    _ref_id = config.get("ref_id", [0])
    # Initialize system state (RNG)
    safe_state(args.quiet)
    is_pbr = args.type == "neilf"
    inpaint(is_pbr, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.finetune_iteration, args.exp_setting)