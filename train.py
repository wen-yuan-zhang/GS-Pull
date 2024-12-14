import argparse
import torch
import numpy as np
import random
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ours: fixed random seed
    # seed = 1
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str, default='sdf',
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf" or "density". ' 
                        'For reconstructing detailed objects centered in the scene with 360° coverage, "density" provides a better foreground mesh. '
                        'For a stronger regularization and a better balance between foreground and background, choose "sdf".')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=10, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    # ours
    parser.add_argument('--dataset_name', default="real360", help='blender, real360, relight3d')
    parser.add_argument('-w', '--white_bg', action='store_true', help="Use white background (default: False)")        # dtu black; tnt white; mipnerf360 white
    parser.add_argument('--output', type=str, default='output/mipnerf360', help='output directory(do not include experiment name')    # TODO: change this
    parser.add_argument('--resolution', type=int, default=1, help='image resolution. (Courthouse 2 especially)')    # TODO: change this

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    
    # ----- Optimize coarse SuGaR -----
    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': args.output,
        'eval': args.eval,
        'estimation_factor': 0.2,
        'normal_factor': 0.0,
        'gpu': args.gpu,
        'dataset_name': args.dataset_name,
        'white_bg': args.white_bg,
        'image_resolution': args.resolution,
    })
    coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
