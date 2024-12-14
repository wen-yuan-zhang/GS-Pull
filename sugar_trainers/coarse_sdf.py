import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
import open3d as o3d
import trimesh
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from pytorch3d.ops import knn_points
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss
from np_utils.train import Runner
from np_utils.dataset import Dataset
from rich.console import Console
import time
import shutil
from tensorboardX import SummaryWriter


def coarse_training_with_sdf_regularization(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    initialize_from_trained_3dgs = True  # True or False
    if initialize_from_trained_3dgs:
        prune_at_start = False
        start_pruning_threshold = 0.5

    n_points_at_start = None  # If None, takes all points in the SfM point cloud

    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 4  

        
    # -----Radiance Mesh-----
    triangle_scale=1.
    
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = False

        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = 15_000

    spatial_lr_scale = None
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
        
    # Densifier and pruning
    heavy_densification = False
    if initialize_from_trained_3dgs:
        densify_from_iter = 500 + 99999 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 - 7000 # 7000
    else:
        densify_from_iter = 500 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 # 7000

    if heavy_densification:
        densification_interval = 50  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
    else:
        densification_interval = 100  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0002  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01

    # Data processing and batching
    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # Regularization
    enforce_entropy_regularization = True
    if enforce_entropy_regularization:
        start_entropy_regularization_from = 7000
        end_entropy_regularization_at = 9000
        entropy_regularization_factor = 0.1
            
    regularize_sdf = True
    if regularize_sdf:
        beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'
        
        start_sdf_regularization_from = 9000
        regularize_sdf_only_for_gaussians_with_high_opacity = False
        if regularize_sdf_only_for_gaussians_with_high_opacity:
            sdf_regularization_opacity_threshold = 0.5
            
        use_sdf_estimation_loss = True
        enforce_samples_to_be_on_surface = False
        if use_sdf_estimation_loss or enforce_samples_to_be_on_surface:
            sdf_estimation_mode = 'sdf'  # 'sdf' or 'density'

            start_sdf_estimation_from = 9000  # 7000
            
            sample_only_in_gaussians_close_to_surface = True
            close_gaussian_threshold = 2.  # 2.
            
            backpropagate_gradients_through_depth = False  # True
            
        use_sdf_better_normal_loss = False
        if use_sdf_better_normal_loss:
            start_sdf_better_normal_from = 9000
            # sdf_better_normal_factor = 0.2  # 0.1 or 0.2?
            sdf_better_normal_gradient_through_normal_only = True
        
        density_factor = 1. / 16. # 1. / 16.
        if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and sdf_estimation_mode == 'density':
            density_factor = 1.
        density_threshold = 1.  # 0.5 * density_factor
        n_samples_for_sdf_regularization = 1_000_000  # 300_000
        sdf_sampling_scale_factor = 1.5
        sdf_sampling_proportional_to_volume = False


    regularity_knn = 16  # 8 until now
    # regularity_knn = 8
    regularity_samples = -1 # Retry with 1000, 10000
    reset_neighbors_every = 500  # 500 until now
    regularize_from = 7000  # 0 until now
    start_reset_neighbors_from = 7000+1  # 0 until now (should be equal to regularize_from + 1?)
    prune_when_starting_regularization = False

        
    # Opacity management
    prune_low_opacity_gaussians_at = [9000]
    prune_hard_opacity_threshold = 0.5

    do_sh_warmup = False
    current_sh_levels = sh_levels = 4   # nerfmodel.gaussians.active_sh_degree + 1
    CONSOLE.print("Changing sh_levels to match the loaded model:", sh_levels)

    # -----Log and save-----
    print_loss_every_n_iterations = 50
    save_model_every_n_iterations = 2000

    # ====================End of parameters====================

    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load    
    

    sugar_checkpoint_path = args.output_dir
    
    use_eval_split = args.eval
    
    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("SUGAR checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Output directory:", args.output_dir)
    CONSOLE.print("Eval split:", use_eval_split)
    CONSOLE.print("---------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    # CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        dataset_name=args.dataset_name,
        image_resolution=args.image_resolution,
        )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # Point cloud
    if initialize_from_trained_3dgs:
        with torch.no_grad():    
            print("Initializing model from trained 3DGS...")
            with torch.no_grad():
                sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))
            
            from sugar_utils.spherical_harmonics import SH2RGB
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
            if prune_at_start:
                with torch.no_grad():
                    start_prune_mask = nerfmodel.gaussians.get_opacity.view(-1) > start_pruning_threshold
                    points = points[start_prune_mask]
                    colors = colors[start_prune_mask]
            n_points = len(points)
    else:
        CONSOLE.print("\nLoading SfM point cloud...")
        pcd = fetchPly(ply_path)
        points = torch.tensor(pcd.points, device=nerfmodel.device).float().cuda()
        colors = torch.tensor(pcd.colors, device=nerfmodel.device).float().cuda()
    
        if n_points_at_start is not None:
            n_points = n_points_at_start
            pts_idx = torch.randperm(len(points))[:n_points]
            points, colors = points.to(device)[pts_idx], colors.to(device)[pts_idx]
        else:
            n_points = len(points)
            
    CONSOLE.print(f"Point cloud generated. Number of points: {len(points)}")
    

    o3d_mesh = None
    learn_surface_mesh_positions = False
    learn_surface_mesh_opacity = False
    learn_surface_mesh_scales = False
    n_gaussians_per_surface_triangle=1
    
    if not regularize_sdf:
        beta_mode = None
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    start_iteration = 7000
    scene_name = args.output_dir.split('/')[-1]
    if start_iteration == 7000:
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=points,      # nerfmodel.gaussians.get_xyz.data,
            colors=colors,      # 0.5 + _C0 * nerfmodel.gaussians.get_features.data[:, 0, :],
            initialize=True,
            sh_levels=sh_levels,
            learnable_positions=learnable_positions,
            triangle_scale=triangle_scale,
            keep_track_of_knn=True,
            knn_to_track=regularity_knn,
            beta_mode=beta_mode,
            freeze_gaussians=freeze_gaussians,
            surface_mesh_to_bind=o3d_mesh,
            surface_mesh_thickness=None,
            learn_surface_mesh_positions=learn_surface_mesh_positions,
            learn_surface_mesh_opacity=learn_surface_mesh_opacity,
            learn_surface_mesh_scales=learn_surface_mesh_scales,
            n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
            )
        if initialize_from_trained_3dgs:
            with torch.no_grad():
                CONSOLE.print("Initializing 3D gaussians from 3D gaussians...")
                if prune_at_start:
                    sugar._scales[...] = nerfmodel.gaussians._scaling.detach()[start_prune_mask]
                    sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()[start_prune_mask]
                    sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()[start_prune_mask]
                    sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()[start_prune_mask]
                    sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()[start_prune_mask]
                else:
                    sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
                    sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
                    sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
                    sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
                    sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
    elif start_iteration > 7000:
        initialize_from_trained_3dgs = False
        checkpoint = torch.load(os.path.join(sugar_checkpoint_path, str(start_iteration)+'.pt'), map_location=nerfmodel.device)
        colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=checkpoint['state_dict']['_points'],
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
        )
        ck_state_dict = checkpoint['state_dict']
        if '_weights' in ck_state_dict:
            del ck_state_dict['_weights']
        sugar.load_state_dict(ck_state_dict, strict=False)
    print(scene_name)
    if scene_name in ['Barn', 'Meetingroom', 'Courthouse']:
        sugar.part_num = 4
    else:
        sugar.part_num = 1
    sugar.neus = Runner(sugar_checkpoint_path, None, part_num=sugar.part_num)

    # ours: log and record
    logger = SummaryWriter(os.path.join(sugar_checkpoint_path, 'logs'))
    record_dir = os.path.join(sugar_checkpoint_path, 'record')
    os.makedirs(record_dir, exist_ok=True)
    shutil.copy('train.py', os.path.join(record_dir, 'train.py'))
    shutil.copytree('sugar_trainers', os.path.join(record_dir, 'sugar_trainers'), dirs_exist_ok = True)
    shutil.copytree('sugar_scene', os.path.join(record_dir, 'sugar_scene'), dirs_exist_ok = True)
    shutil.copytree('np_utils', os.path.join(record_dir, 'np_utils'), dirs_exist_ok = True)

    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()
    
    
    # ====================Initialize optimizer====================
    if spatial_lr_scale is None:
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])

        
    # ====================Initialize densifier====================
    gaussian_densifier = SuGaRDensifier(
        sugar_model=sugar,
        sugar_optimizer=optimizer,
        max_grad=densify_grad_threshold,
        min_opacity=prune_opacity_threshold,
        max_screen_size=densify_screen_size_threshold,
        scene_extent=cameras_spatial_extent,
        percent_dense=densification_percent_distinction,
        )
    CONSOLE.print("Densifier initialized.")
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')

    # ours init
    if start_iteration == 9000:
        print("Prunning Pointcloud Using Opacity...")
        prune_mask = (gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
        gaussian_densifier.prune_points(prune_mask)
        print('After Prunning: {} Gaussians Left.'.format(sugar.points.shape[0]))
        sugar.visual_point_cloud(9001, sugar_checkpoint_path)

    sugar.reset_neighbors()

    if start_iteration == 9000:
        sugar.neus.reset_datasets(sugar_checkpoint_path, sugar.points.detach().cpu().numpy(), iteration=9000, scene_name=scene_name)


    # ====================Start training====================
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
    
    if initialize_from_trained_3dgs:
        iteration = 7000 - 1
    else:
        iteration = start_iteration
    
    iteration += 1
    _iter = iteration
    train_render = False
    train_scaling = False
    train_sdf = False
    train_normal = False
    has_not_pruned = True
    last_resample_iteration = last_save_iteration = last_reset_iteration = last_visual_iteration = -1
    cur_part_num = sugar.part_num+1
    evaluated_mesh = None
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)

        for i in range(0, train_num_images, train_num_images_per_batch):
            # iteration += 1
            if iteration < 9001:
                iteration += 1
                _iter += 1
                train_render = True
                train_scaling = False
                train_sdf = False
                train_normal = False
            else:
                # splat every 10 pulling iter
                if _iter % 10 == 0:
                    iteration += 1
                    _iter += 1
                    train_render = True
                    train_scaling = False
                    train_sdf = True
                    train_normal = True
                else:
                    if cur_part_num >= sugar.part_num:     # visit all parts
                        _iter += 1
                        cur_part_num = 1
                    else:
                        cur_part_num += 1
                    train_render = False
                    train_scaling = False
                    train_sdf = True
                    train_normal = True

            # Update learning rates
            if iteration >= 9000:
                udfnet_lr = sugar.neus.get_learning_rate_at_iteration(_iter - 9000)
            else:
                udfnet_lr = 0.
            optimizer.update_learning_rate(iteration, sdfnet_lr=udfnet_lr)

            if iteration == 9001 and has_not_pruned:
                print("Prunning Pointcloud Using Opacity...")
                prune_mask = (gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
                gaussian_densifier.prune_points(prune_mask)
                print('After Prunning: {} Gaussians Left.'.format(sugar.points.shape[0]))
                sugar.visual_point_cloud(iteration=9001, checkpoint_path=sugar_checkpoint_path)
                sugar.neus.reset_datasets(sugar_checkpoint_path, sugar.points.detach().cpu().numpy(), iteration=9000, scene_name=scene_name)
                has_not_pruned = False

            if iteration % 1000 == 0 and iteration != last_resample_iteration and iteration > 9500:
                if iteration % 2000 == 0:
                    print('Recalculating Sample Points...')
                    sugar.neus.reset_datasets(sugar_checkpoint_path, sugar.points.detach().cpu().numpy(), iteration=iteration, scene_name=scene_name)
                last_resample_iteration = iteration


            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            
            # Computing rgb predictions
            loss = render_loss = opacity_loss = scaling_loss = sdf_loss = normal_loss = 0
            pred_rgb = None
            if train_render:
                outputs = sugar.render_image_gaussian_rasterizer(
                    camera_indices=camera_indices.item(),
                    verbose=False,
                    bg_color=torch.Tensor([1.0, 1.0, 1.0]).to(sugar.device) if args.white_bg else None,
                    sh_deg=current_sh_levels-1,
                    sh_rotations=None,
                    compute_color_in_rasterizer=compute_color_in_rasterizer,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                    return_opacities=enforce_entropy_regularization,
                    use_pulled=False,
                    )
                pred_rgb = outputs['image'].view(-1,
                    sugar.image_height,
                    sugar.image_width,
                    3)
                radii = outputs['radii']
                viewspace_points = outputs['viewspace_points']
                if enforce_entropy_regularization:
                    opacities = outputs['opacities']

                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)

                # Gather rgb ground truth
                gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)
                gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)

                # Compute loss
                loss = loss_fn(pred_rgb, gt_rgb)
                render_loss = loss.item()

            if enforce_entropy_regularization and iteration > start_entropy_regularization_from and iteration < end_entropy_regularization_at:
                if iteration == start_entropy_regularization_from + 1:
                    CONSOLE.print("\n---INFO---\nStarting entropy regularization.")
                if iteration == end_entropy_regularization_at - 1:
                    CONSOLE.print("\n---INFO---\nStopping entropy regularization.")
                visibility_filter = radii > 0
                if visibility_filter is not None:
                    vis_opacities = opacities[visibility_filter]
                else:
                    vis_opacities = opacities

                opacity_loss = (1-sugar.strengths).abs().mean()
                opacity_loss = (
                    - vis_opacities * torch.log(vis_opacities + 1e-10)
                    - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                    ).mean()
                loss = loss + entropy_regularization_factor * opacity_loss

            # if iteration == regularize_from:
            #     CONSOLE.print("Starting regularization...")
            if iteration > start_sdf_regularization_from:
                if (iteration >= start_reset_neighbors_from) and ((iteration == start_sdf_regularization_from + 1) or (iteration % reset_neighbors_every == 0)) and iteration != last_reset_iteration:
                    CONSOLE.print("\n---INFO---\nResetting neighbors...")
                    sugar.reset_neighbors()
                    last_reset_iteration = iteration
                # -----------------------------------------------
                # scaling loss
                if train_scaling:
                    # scaling_loss = torch.abs(sugar.scaling.min(1)[0] - 1e-7).mean()
                    clamped_scaling = torch.clamp(sugar.scaling.min(1)[0], min=1e-4)
                    scaling_loss = torch.abs(clamped_scaling - 1e-4).mean()
                    loss = loss + 100 * scaling_loss
                # pulling loss
                if train_sdf:
                    sugar_points = sugar.points
                    dataset = getattr(sugar.neus, 'dataset'+str(cur_part_num))
                    points, samples, point_gt, points_idx = dataset.get_train_data(10000)

                    samples.requires_grad = True
                    sdf_network = getattr(sugar.neus, 'sdf_network'+str(cur_part_num))
                    gradients_sample = sdf_network.gradient(samples).squeeze()  # 5000x3
                    udf_sample = sdf_network.sdf(samples)  # 5000x1
                    grad_norm = F.normalize(gradients_sample, dim=1)  # 5000x3
                    sample_moved = samples - grad_norm * udf_sample  # 5000x3

                    sdf_loss1 = sugar.neus.ChamferDisL1(points.unsqueeze(0), sample_moved.unsqueeze(0))

                    scaled_sample_moved = sample_moved * dataset.shape_scale + dataset.shape_center
                    knn = knn_points(sample_moved[None], points[None], K=1)
                    knn_idx = knn.idx[0,:,0]
                    # gaussian_inv_scaled_rotation = sugar.get_covariance(
                    #     return_full_matrix=True, return_sqrt=True, inverse_scales=True, scaling_factor=-1, enlarge_minaxis=-1)
                    gaussian_inv_scaled_rotation = sugar.get_covariance(
                        return_full_matrix=True, return_sqrt=True, inverse_scales=True, scaling_factor=100,enlarge_minaxis=100)
                    batch_selected_idx = torch.arange(sugar.points.shape[0],device='cuda')[dataset.part_select_idx][dataset.downsample_idx][points_idx][knn_idx]
                    closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[batch_selected_idx].detach()
                    surf_points = points[knn_idx].detach().clone() * dataset.shape_scale + dataset.shape_center

                    shift = (scaled_sample_moved - surf_points)
                    warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
                    neighbor_opacities = torch.exp(-1. / 2 * neighbor_opacities)
                    if iteration > 10000:
                        sdf_loss2 = torch.abs(1-neighbor_opacities)[neighbor_opacities>0.9].mean()
                    else:
                        sdf_loss2 = 0.
                    sdf_loss = 1.0 * sdf_loss1 + 1.0 * sdf_loss2

                    # ours: pull gs
                    if iteration > 10000:   # delay for very large scene
                        rescaled_sugar_points = (surf_points - dataset.shape_center) / dataset.shape_scale
                        _gradients_sample = sdf_network.gradient(rescaled_sugar_points).squeeze()
                        _udf_sample = sdf_network.sdf(rescaled_sugar_points)
                        _grad_norm = F.normalize(_gradients_sample, dim=1)  #
                        rescaled_sugar_points_moved = rescaled_sugar_points - _grad_norm * _udf_sample
                        sugar_points_moved = rescaled_sugar_points_moved * dataset.shape_scale + dataset.shape_center
                        sugar_points_diff = torch.norm(sugar.points[batch_selected_idx] - sugar_points_moved.detach(), p=2, dim=-1).mean()
                        sdf_loss = sdf_loss + 0.05 * sugar_points_diff

                    loss = loss + sdf_loss
                # norm consistency
                if train_normal:
                    assert train_sdf, 'require train_sdf=True for train_normal!'
                    if iteration > 10000:   # delay for very large scene
                        sugar_normals = sugar.get_normals()[batch_selected_idx]
                        surf_normals = _grad_norm.detach()
                        sugar_normal_loss = torch.abs(torch.sum(surf_normals * sugar_normals, -1).abs() - 1).mean()

                        if iteration > 12000:   # delay for very large scene
                            gaussian_center_normals = sugar_normals.detach()
                            query_normal_loss = torch.abs(torch.sum(grad_norm * gaussian_center_normals, -1).abs() - 1).mean()
                        else:
                            query_normal_loss = 0.

                        normal_loss = 0.1 * sugar_normal_loss + 0.01 * query_normal_loss

                        loss = loss + normal_loss

            # Update parameters
            loss.backward()

            # if iteration % 2 == 0:
            if _iter % 1 == 0 and render_loss != 0:
                logger.add_scalar('Loss/render_loss', render_loss, global_step=iteration)
                logger.add_scalar('Loss/opacity_loss', opacity_loss, global_step=iteration)
                logger.add_scalar('Loss/scaling_loss', scaling_loss, global_step=iteration)
                logger.add_scalar('Loss/sdf_loss', sdf_loss, global_step=iteration)
                logger.add_scalar('Loss/norm_loss', normal_loss, global_step=iteration)
            # visualize mesh
            if (iteration % 500 == 0 and iteration != last_visual_iteration and iteration > 9000) or iteration == num_iterations:
                with torch.no_grad():
                    vertices_list = []
                    triangles_list = []
                    for part in range(1, sugar.part_num + 1):
                        evaluated_mesh = sugar.marching_cubes_part(iteration, sugar_checkpoint_path, vertex_color=True,
                                                                   thres=0.002, part=part)
                        vertices, triangles = evaluated_mesh.vertices, evaluated_mesh.faces
                        vertices_list.append(vertices)
                        triangles_list.append(triangles)
                    # Merge all meshes
                    all_vertices = np.vstack(vertices_list)
                    all_triangles = np.vstack([
                        tri + sum(map(len, vertices_list[:i]))
                        for i, tri in enumerate(triangles_list)
                    ])
                    combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_triangles)
                    combined_mesh.export(os.path.join(sugar_checkpoint_path, 'meshes', 'mcubes_merge_{}.ply'.format(iteration)))

                    sugar.visual_point_cloud(iteration, sugar_checkpoint_path)
                    sugar.validate_image(pred_rgb, camera_indices.item(), iteration, sugar_checkpoint_path, )
                    sugar.validate_normal_image(1, iteration, sugar_checkpoint_path)
                last_visual_iteration = iteration
                torch.cuda.empty_cache()


            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if _iter % 50 == 0:
                print(iteration, _iter, 'loss:', loss.item(), 'udflr:', udfnet_lr)
                
            # Save model
            if iteration % save_model_every_n_iterations == 0 and iteration != last_save_iteration:
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                sugar.save_ply(sugar_checkpoint_path, iteration)
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")
                if iteration > 9000:
                    sugar.neus.save_checkpoint(sugar_checkpoint_path, iteration)
                last_save_iteration = iteration

            if iteration >= num_iterations:
                break

        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )
    sugar.save_ply(sugar_checkpoint_path, iteration)
    sugar.neus.save_checkpoint(sugar_checkpoint_path, iteration)

    CONSOLE.print("Final model saved.")
    return model_path