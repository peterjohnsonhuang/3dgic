# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import numpy as np
from PIL import Image
import colorsys
import json

import cv2
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull, Delaunay
from scipy import ndimage
from skimage.morphology import convex_hull_image
from utils.ggutils import visualize_obj, feature_to_rgb
import copy

from scene.gamma_trans import LearningGammaTransform
from scene.derect_light_sh import DirectLightEnv

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask

def removal_setup(is_pbr, opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        print(gaussians._objects_dc.shape)
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        # print(prob_obj3d.max(0)[0].shape)
        # input()
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        # print(removal_thresh)
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)

        mask3d = mask3d.float()[:,None,None]

    # fix some gaussians
    # print(mask3d)
    import copy
    removed_gaussian = copy.copy(gaussians)
    removed_gaussian.removal_setup(is_pbr, opt,1-mask3d)
    gaussians.removal_setup(is_pbr, opt,mask3d)
    
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians, removed_gaussian

# from edit_object_inpaint import depth_image_to_point_cloud, world_to_new_view, get_init_points
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


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
    
    # print(inpaint_mask2d)
    # print(source_depths, source_intrinsics, source_c2ws, img_wh, source_image, inpaint_mask2d)
    # print(source_depths.shape)
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
        # print(mask.shape)

        dirs = torch.stack(
        [
            (xs - source_intrinsics[num][0, 2]) / source_intrinsics[num][0, 0],
            (ys - source_intrinsics[num][1, 2]) / source_intrinsics[num][1, 1],
            torch.ones_like(xs),
        ],
        -1,
        )
        # print(dirs.shape, source_c2ws.shape)
        # print(source_c2ws[num][:3, :3])
        # print(dirs)
        # print(dirs.max(), dirs.min())
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




def world_to_new_view(points, target_intrinsics, target_w2cs):
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

    
    return torch.cat((uv.T, points[:,3:]),1)

def get_init_points(cam, image,mask,default_depth=False):
            
    # print(cam.image_name)
    # print(cam.depths.shape)
    
    
    depth = (cam.depths.clone()).float().permute(2,0,1)
    
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
    mask2d = (mask >128).to(depth.device).squeeze()
    # print(mask2d.nonzero())
    # print(mask2d.shape)
    # input()
    # print(viewpoint_cam.objects)
    # depth_point = depth[mask2d>0]
    # mask_index = mask2d.nonzero(as_tuple=True)
    # print(mask_index)
    # mask_index = mask2d.reshape(-1).nonzero()
    # print(len(mask_index[0]), depth[mask_index].shape)
    # mask_depth = depth[mask_index]
    gt_image = image.cuda()
    # print(gt_image)
    # exit()
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

    # print(depth.shape, A.device, extrinsic.device, gt_image.shape, mask2d.shape)
    # input()
    points = depth_image_to_point_cloud(depth, A, extrinsic, depth.shape, gt_image, mask2d)
    # print(points.shape)
    # input()
    # points_all = depth_image_to_point_cloud(depth, A, extrinsic, depth.shape, gt_image, torch.ones_like(mask2d))
    # points_all = points
    # print(points.shape)
    # input()
    # points = points[mask_index].squeeze(1)
    
    
    # del depth, gt_image
    return points

def find_intersect_mask(mask_input, viewpoint_cam, views, select_obj_id, id_now, original_gaussians, is_pbr, pipeline, background, opt, pbr_kwargs, classifier):
    all_mask_stack=[]
    mask_now = mask_input.numpy()
    mask_now = torch.Tensor(convex_hull_image(mask_now)).cuda()

    gt_shape = mask_now.shape
    gt_image_now = viewpoint_cam.original_image.cuda() * (1-(mask_input.cuda().unsqueeze(0).repeat(3,1,1))/255.0)
    depth_remove_now = viewpoint_cam.depths[:,:,1]
    # print(depth_remove_now.max(), depth_remove_now.min())
    if is_pbr:
        render = render_fn_dict["neilf"]
    else:
        render = render_fn_dict["render"]
    # print(mask_now.nonzero())
    
    # mask_last = mask_now
    
    # print(gt_image.shape)
    id_list = torch.multinomial(torch.ones(len(views)).float(), 30)
    # print(id_list)
    # exit()
    
    for idx, view in enumerate(tqdm(views.copy(), desc="intersect mask")):
        if ((idx < int(id_now+30)) and (idx> (int(id_now)-30))):
        # if idx in id_list:
            if view.objects is None:
                if is_pbr:
                    original_results =  render(view, original_gaussians, pipeline, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
                else:
                    original_results =  render(view, original_gaussians, pipeline, background)
                original_rendering_obj = original_results["render_object"]
                ori_logits = classifier(original_rendering_obj)
                ori_pred_obj = torch.argmax(ori_logits,dim=0)
                gt_object_others = ori_pred_obj.squeeze(0)
            else:
                gt_object_others = view.objects.squeeze(0)
            # mask_others = (((gt_object_others * (gt_object_others == select_obj_id[0])) > 0).int()*255.0).cuda()
            mask_fg = (((gt_object_others * (gt_object_others == select_obj_id[0])) > 0).int()).numpy()
            # print(mask_fg.shape)
            mask_fg = torch.Tensor(convex_hull_image(mask_fg)).cuda()

            mask_others = 1 - mask_fg
            # view.objects = mask_others.cuda()*255.0
            # print(view.original_image.shape, mask_others.shape)

            # view.original_image = (view.original_image.to(mask_others.device)) * ((mask_others).unsqueeze(0).repeat(3,1,1))
            # view.depths = torch.ones_like(view.original_image).cuda()
            point = get_init_points(view, image=(view.original_image.to(mask_others.device)) * ((mask_others).unsqueeze(0).repeat(3,1,1)), mask=mask_others.cuda()*255.0)

            # input()

            depth_remove = view.depths[:,:,1].clone()

            depth_point = get_init_points(view, image=(depth_remove.unsqueeze(0).repeat(3,1,1).to(mask_others.device)), mask=mask_others.cuda()*255.0)
            # print(point.shape)
            # print(depth_point.min(), depth_point.max())

            R,T,FovY,FovX, width, height = viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.FoVy, viewpoint_cam.FoVx, viewpoint_cam.image_width, viewpoint_cam.image_height
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
                            [0,0,1.0]]).to(point.device)
            # print(mask_index[0].shape)
            extrinsic = viewpoint_cam.camera_back_project
            wtc_project = viewpoint_cam.world_view_transform.T
            target_point_ref = world_to_new_view(point, A, torch.Tensor(wtc_project).to(point.device))

            target_depth_ref = world_to_new_view(depth_point, A, torch.Tensor(wtc_project).to(point.device))

            

            # print(target_point_ref.shape)

            gt_image = torch.zeros_like(view.original_image)
            gt_depth = torch.ones_like(view.original_image)*10000.0
            # print(gt_image.shape)
            b1, b2, b3, b4 = (target_point_ref[:,0]<=(gt_image.shape[2]-1)), (target_point_ref[:,0]>=0), (target_point_ref[:,1]<=(gt_image.shape[1]-1)), (target_point_ref[:,1]>=0)


            # print(b1.shape)
            # print((b1*b2*b3*b4).nonzero(as_tuple=True))

            selected = (b1*b2*b3*b4).nonzero(as_tuple=True)[0]

            target_point_ref_clone = target_point_ref.detach().clone()[selected]
            target_depth_ref_clone = target_depth_ref.detach().clone()[selected]

            swap_index = target_point_ref_clone.clone().long()[:,:2].T ### find which pixels to swap first


            ### only preserve pixels where depth is smaller than depth_remove_now
            gt_depth = gt_depth.permute(1,2,0)
            gt_depth[(swap_index[1], swap_index[0])] = target_depth_ref_clone[:,2:]
            # print(target_depth_ref_clone[:,2:])
            # exit()
            gt_depth = gt_depth[:,:,0]

            depth_mask = torch.ge(depth_remove_now, gt_depth).int() ### where the projected depth is smaller than current removed depth (which means this pixel is already observed by another view)

            # depth_mask  = torch.ones_like(depth_mask) ### delete this line if penetration of depth exist

            # print(depth_remove_now)
            # print(gt_depth)
            # exit()
            # print(swap_index)
            # print(depth_mask.shape)

            ## find target ref pixels where indices already in swap_index

            # input()


            gt_image = gt_image.permute(1,2,0)
            
            # print(target_point_ref_clone)
            # print(target_point_ref_clone.shape)
            gt_image[(swap_index[1], swap_index[0])] = target_point_ref_clone[:,2:]

            gt_image = gt_image * (depth_mask.unsqueeze(-1).repeat(1,1,3))
            
            # print(gt_depth.shape)
            
            # input()
            # gt_image[]
            # print(gt_swap.shape)
            # print(gt_image.shape)
            # print(gt_image.nonzero())
            # Image.fromarray(gt_image[:,:,0].int().cpu().numpy()).save("test_mask.png")
            
            gt_image = gt_image.permute(2,0,1)
            # print(gt_image_now.max(), gt_image.max())
            # print(mask_now.max())
            
            gt_image = gt_image * ((mask_now.float().unsqueeze(0).repeat(3,1,1)/255.0).float())
            gt_image_now = torch.maximum(gt_image_now, gt_image)
            # print(gt_image_now.max(), gt_image.max())
            # exit()
            # mask_last = mask_now
            # print()
            swap_index = depth_mask.nonzero(as_tuple=True)
            mask_now[(swap_index[0], swap_index[1])] = torch.zeros(len(swap_index[1])).to(mask_now.device)
            # # input()
            # print(gt_image_now.shape, mask_now.shape)
            # input()
    #         torchvision.utils.save_image(gt_image_now.cpu(), "./mask_demo_counter/test_seen_image_torch_{}.png".format(idx))
    #         torchvision.utils.save_image(mask_now.cpu().unsqueeze(0).repeat(3,1,1), "./mask_demo_counter/test_mask_torch_{}.png".format(idx))
    #         # input()
        


    # fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    # size = (gt_shape[-1]*2,gt_shape[-2])
    # fps = float(20)
    # writer = cv2.VideoWriter("./mask_demo_counter/mask_demo.mp4", fourcc, fps, size)
    # for i in range(idx +1):
    #     image = np.array(Image.open("./mask_demo_counter/test_seen_image_torch_{}.png".format(i)))
    #     mask = np.array(Image.open("./mask_demo_counter/test_mask_torch_{}.png".format(i)))
    #     result = np.hstack([image,mask])
    #     result = result.astype('uint8')
    #     writer.write(result[:,:,::-1])

    # writer.release()

    # exit()
    mask_now = mask_now.cpu().numpy()
    # struct1 = ndimage.generate_binary_structure(2, 2)
    struct1 = np.ones((25,25))
    struct2 = np.ones((20,20))
    struct3 = np.ones((10,10))
    # mask_now = ndimage.binary_erosion(ndimage.binary_dilation(ndimage.binary_erosion(mask_now, struct1), struct2))
    # mask_now = ndimage.binary_erosion(ndimage.binary_dilation(mask_now, struct2), struct1)
    # mask_now = ndimage.binary_dilation(mask_now,struct3)
    mask_now = ndimage.binary_fill_holes(mask_now)
    # mask_now = convex_hull_image(mask_now)
    # torchvision.utils.save_image(torch.Tensor(mask_now).unsqueeze(0).repeat(3,1,1), "./test_mask_fix.png")

    return mask_now
        




def render_set(is_pbr, pbr_kwargs, model_path, name, iteration, views, gaussians, pipeline, background, classifier, select_obj_id=0, render_intersect=False):
    if is_pbr:
        render = render_fn_dict["neilf"]
    else:
        render = render_fn_dict["render"]
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    ori_render_path = os.path.join(model_path, name, "ours{}".format(iteration), "ori_renders")
    depth_path = os.path.join(model_path, name, "ours{}".format(iteration), "depth")
    depth_real_path = os.path.join(model_path, name, "ours{}".format(iteration), "depth_removal")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    gt_mask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    remove_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "remove_objects_pred")
    remain_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "remain_objects_pred")
    inpaint_mask_path = os.path.join(model_path, name, "ours{}".format(iteration), "inpaint_mask_pred")
    removed_base_color_path = os.path.join(model_path, name, "ours{}".format(iteration), "base_color")
    pbr_path = os.path.join(model_path, name, "ours{}".format(iteration), "pbr")
    # opacity_path = os.path.join(model_path, name, "ours{}".format(iteration), "opacity")
    makedirs(render_path, exist_ok=True)
    makedirs(ori_render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(gt_mask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_real_path, exist_ok=True)
    makedirs(remove_obj_path, exist_ok=True)
    makedirs(remain_obj_path, exist_ok=True)
    makedirs(inpaint_mask_path, exist_ok=True)
    makedirs(removed_base_color_path, exist_ok=True)
    makedirs(pbr_path, exist_ok=True)
    gt_shape = -1

    gaussians, removed_gaussian, original_gaussians = gaussians
    # print(len(views))
    # input()

    with torch.no_grad():

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # print(is_pbr)
            if is_pbr:
                results = render(view, gaussians, pipeline, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
                removed_results = render(view, removed_gaussian, pipeline, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
                original_results =  render(view, original_gaussians, pipeline, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
            else:
                results = render(view, gaussians, pipeline, background)
            # results = render(view, gaussians, pipeline, background)
                removed_results = render(view, removed_gaussian, pipeline, background)
                original_results =  render(view, original_gaussians, pipeline, background)

            rendering = results["render"]
            rendering_obj = results["render_object"]

            original_rendering = original_results["render"]
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


            removed_rendering_obj = removed_results["render_object"]
            original_rendering_obj = original_results["render_object"]

            depth_real = results["depth"]
            original_depth = original_results["depth"]

            logits = classifier(rendering_obj)
            pred_obj = torch.argmax(logits,dim=0)
            pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

            ori_logits = classifier(original_rendering_obj)
            ori_pred_obj = torch.argmax(ori_logits,dim=0)
            ori_pred_obj_mask = visualize_obj(ori_pred_obj.cpu())
            

            logits = classifier(removed_rendering_obj.cuda())
            pred_obj = torch.argmax(logits,dim=0).cpu()
            # pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

            if view.objects is None:
                # view.objects = ori_pred_obj
                gt_objects = ori_pred_obj.squeeze(0)
            else:
                gt_objects = view.objects.squeeze(0)
            remove_obj = ((gt_objects * (gt_objects == select_obj_id[0])) > 0).int().cpu()*255.0
            # remain_obj = (gt_objects * (~(gt_objects == select_obj_id[0]))).cpu().int().numpy().astype(np.uint8)
            remain_obj = (gt_objects).cpu().int().numpy().astype(np.uint8)
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
            gt_objects = gt_objects.cpu().int().numpy().astype(np.uint8)

            if render_intersect:
                intersect_mask = find_intersect_mask(remove_obj.clone(), copy.deepcopy(view), views.copy(), select_obj_id, id_now = idx, original_gaussians = original_gaussians, is_pbr = is_pbr, pipeline=pipeline, background=background, opt=opt, pbr_kwargs=pbr_kwargs, classifier=classifier)

            # print(rendered_depth.shape)
            # print(depth_real.max(), depth_real.min())
            # input()

            rgb_mask = feature_to_rgb(rendering_obj)
            if render_intersect:
                Image.fromarray(intersect_mask).save(os.path.join(inpaint_mask_path, '{0:05d}'.format(idx) + ".png"))
            torch.save(original_depth,os.path.join(depth_path, '{0:05d}_original'.format(idx) + ".pt"))
            torch.save(depth_real,os.path.join(depth_path, '{0:05d}_remove'.format(idx) + ".pt"))
            Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(gt_objects).save(os.path.join(gt_mask_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
            # Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(remain_obj).save(os.path.join(remain_obj_path, '{0:05d}'.format(idx) + ".png"))
            gt = view.original_image[0:3, :, :]
            gt_shape = gt.shape

            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(remove_obj, os.path.join(remove_obj_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(original_rendering, os.path.join(ori_render_path, '{0:05d}'.format(idx) + ".png"))
            if is_pbr:
                torchvision.utils.save_image(base_color, os.path.join(removed_base_color_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(pbr, os.path.join(pbr_path, '{0:05d}'.format(idx) + ".png"))


            idx+=1
            torchvision.utils.save_image(rendering, os.path.join(render_path, 'frame_{0:05d}'.format(idx) + ".png"))
            if is_pbr:
                torchvision.utils.save_image(base_color, os.path.join(removed_base_color_path, 'frame_{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(pbr, os.path.join(pbr_path, 'frame_{0:05d}'.format(idx) + ".png"))
            if render_intersect:
                Image.fromarray(intersect_mask).save(os.path.join(inpaint_mask_path, 'frame_{0:05d}'.format(idx) + ".png"))
            

            d_min, d_max = depth_real.min(), depth_real.min()+5
            # d_min, d_max = depth_real.min(), depth_real.max()

            # print(d_min, d_max)
            # input()
            depth_real = (depth_real.float().clamp(0,d_max) - d_min)/(d_max-d_min)
            torchvision.utils.save_image(depth_real, os.path.join(depth_real_path, '{0:05d}'.format(idx-1) + ".png"))
            torch.save((d_min,d_max),os.path.join(depth_real_path, '{0:05d}_range'.format(idx-1) + ".pt"))

            # d_min_ori, d_max_ori = original_depth.min(), original_depth.min()+5
            d_min_ori, d_max_ori = original_depth.min(), original_depth.max()

            # print(d_min, d_max)
            # input()
            original_depth = (original_depth.float().clamp(0,d_max_ori) - d_min_ori)/(d_max_ori-d_min_ori)
            torchvision.utils.save_image(original_depth, os.path.join(depth_path, '{0:05d}'.format(idx-1) + ".png"))
            torch.save((d_min_ori,d_max_ori),os.path.join(depth_path, '{0:05d}_range'.format(idx-1) + ".pt"))

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


def removal(is_pbr, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float, render_intersect: bool):
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
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    from utils.system_utils import searchForMaxIteration
    scene.loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    # print(loaded_iter)
    # input()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
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


    original_gaussians = copy.deepcopy(gaussians)



    # 2. remove selected object
    gaussians, removed_gaussian = removal_setup(is_pbr, opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh)
    load_iter = copy.deepcopy(scene.loaded_iter)
    del scene
    torch.cuda.empty_cache()
    # 3. render new result
    scene = Scene(dataset, gaussians, load_iteration='_object_removal/iteration_'+str(load_iter), shuffle=False)
    with torch.no_grad():
        if not skip_train:
             render_set(is_pbr, pbr_kwargs,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), [gaussians, removed_gaussian, original_gaussians], pipeline, background, classifier, select_obj_id, render_intersect)

        if not skip_test and len(scene.getTestCameras()) != 0:
             render_set(is_pbr, pbr_kwargs,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), [gaussians, removed_gaussian, original_gaussians], pipeline, background, classifier, select_obj_id, render_intersect)

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
    parser.add_argument("--render_intersect", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')

    parser.add_argument("--config_file", type=str, default="configs/object_removal/bear.json", help="Path to the configuration file")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)



    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # print(args.render_intersect)

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
    args.source_path = config.get("source_path", "./data/bear")
    # args.checkpoint = config.get("checkpoint", None)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    is_pbr = args.type == "neilf"
    # print(args.type)
    # input()
    removal(is_pbr, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.render_intersect)


