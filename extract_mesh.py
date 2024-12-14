
from np_utils.train import Runner
import os
import torch
import numpy as np
import trimesh
from np_utils.extract_mesh_meshudf import get_mesh_udf_fast
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import mcubes


def clean_mesh(mesh, source_path, gs_output_path, dataset_name='real360'):
    from sugar_scene.cameras import CamerasWrapper, load_gs_cameras
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh import rasterizer
    from pytorch3d.renderer.cameras import PerspectiveCameras
    cam_list = load_gs_cameras(
        source_path=source_path,
        gs_output_path=gs_output_path,
        load_gt_images=False,
        dataset_name=dataset_name,
    )
    training_cameras = CamerasWrapper(cam_list)
    image_height, image_width = training_cameras.gs_cameras[0].image_height, training_cameras.gs_cameras[0].image_width

    num_faces = len(mesh.faces)
    nb_visible = 1
    count = torch.zeros(num_faces, device="cuda")
    # K, R, t, sizes = cams[:4]

    n = len(training_cameras.gs_cameras)
    with torch.no_grad():
        for i in tqdm(range(n), desc="clean_faces"):
            vertices = torch.from_numpy(mesh.vertices).cuda().float()
            faces = torch.from_numpy(mesh.faces).cuda().long()
            meshes = Meshes(verts=[vertices],
                            faces=[faces])
            raster_settings = rasterizer.RasterizationSettings(image_size=(image_height, image_width),
                                                               faces_per_pixel=1)
            meshRasterizer = rasterizer.MeshRasterizer(training_cameras.p3d_cameras[i], raster_settings)

            with torch.no_grad():
                ret = meshRasterizer(meshes)
                pix_to_face = ret.pix_to_face
                # pix_to_face, zbuf, bar, pixd =

            visible_faces = pix_to_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1

    pred_visible_mask = (count >= nb_visible).cpu()

    mesh.update_faces(pred_visible_mask)
    return mesh


def marching_cubes_udf_part(neus, iteration=0, checkpoint_path='.', resolution=700, vertex_color=False, thres=5.,
                            part=1, world_space=True, crop_border=True):
    dataset = getattr(neus, 'dataset' + str(part))
    sdf_network = getattr(neus, 'sdf_network' + str(part))

    # func = sdf_network.sdf
    def func(xyz):
        return torch.abs(sdf_network.sdf(xyz))

    def func_grad(xyz):
        gradients = sdf_network.gradient(xyz)
        gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
        gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector
        return gradients_norm
    dataset.object_bbox_min = np.array([-0.6,-0.6,-0.6])
    dataset.object_bbox_max = np.array([0.6,0.6,0.6])
    voxel_origin = np.ones(3) * dataset.object_bbox_min.min()
    cube_size = dataset.object_bbox_max.max() - dataset.object_bbox_min.min()
    try:
        pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                        indices=None, N_MC=resolution,
                                                                        gradient=True, eps=0.005,
                                                                        border_gradients=True,
                                                                        smooth_borders=True,
                                                                        dist_threshold_ratio=thres,
                                                                        voxel_origin=voxel_origin,
                                                                        cube_size=cube_size)
    except:
        pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                        indices=None, N_MC=resolution,
                                                                        gradient=True, eps=0.005,
                                                                        border_gradients=False,
                                                                        smooth_borders=False,
                                                                        dist_threshold_ratio=thres,
                                                                        voxel_origin=voxel_origin,
                                                                        cube_size=cube_size)

    vertices, triangles = pred_mesh.vertices, pred_mesh.faces

    if vertex_color:
        colors = []
        for pts in torch.tensor(vertices, dtype=torch.float).cuda().split(50000):
            with torch.enable_grad():
                normals = sdf_network.gradient(pts).squeeze()
                normals = F.normalize(normals, p=2, dim=-1)
            ndc_coords = torch.tensor([[0, 0, -1.0]]).cuda()
            nerf_cameras = nerfmodel.training_cameras
            p3d_camera = nerf_cameras.p3d_cameras[3]
            world_coords = p3d_camera.unproject_points(ndc_coords, world_coordinates=True)
            camera_center = p3d_camera.get_camera_center()
            ray_directions = world_coords - camera_center
            ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
            normals = normals * torch.sign((normals * ray_directions).sum(dim=-1, keepdim=True))
            normals = torch.clamp((normals + 1) / 2, 0, 1.)
            colors.append(normals.detach().cpu().numpy() * 255)
        colors = np.concatenate(colors, 0)
    else:
        colors = None

    if world_space:
        vertices = vertices * dataset.shape_scale.cpu().numpy() + dataset.shape_center.cpu().numpy()
        if crop_border:
            vertex_mask = np.all((vertices >= dataset.block_min - dataset.split_size / 40) &
                                 (vertices <= dataset.block_max + dataset.split_size / 40), axis=1)
            faces_mask = np.all(vertex_mask[triangles], axis=1)
            new_faces = triangles[faces_mask]
            new_indices = np.zeros(len(vertex_mask), dtype=int)
            new_indices[vertex_mask] = np.arange(np.sum(vertex_mask))
            triangles = new_indices[new_faces]
            vertices = vertices[vertex_mask]
            if vertex_color:
                colors = colors[vertex_mask]

    debug_path = os.path.join(checkpoint_path, 'meshes')
    os.makedirs(debug_path, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)
    mesh.export(os.path.join(debug_path, 'meshudf_{}_{}.ply'.format(iteration, part)))
    print('UDF Marching Cubes OK.')
    return mesh

def marching_cubes_sdf_part(neus, iteration=0, checkpoint_path='.', resolution=256, vertex_color=False, thres=0.002, part=1, world_space=True, crop_border=True, move_surf=False):
    dataset = getattr(neus, 'dataset' + str(part))
    sdf_network = getattr(neus, 'sdf_network' + str(part))
    dataset.object_bbox_min = np.array([-0.6,-0.6,-0.6])
    dataset.object_bbox_max = np.array([0.6,0.6,0.6])
    bound_min = torch.tensor(dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(dataset.object_bbox_max, dtype=torch.float32)
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                    dim=-1).cuda()  # [N,3]
                    sample_sdf = sdf_network.sdf(pts)  # [N]

                    val = sample_sdf.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()

                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    vertices, triangles = mcubes.marching_cubes(u, thres)
    b_max_np = bound_max.cpu().numpy()
    b_min_np = bound_min.cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    if vertex_color:
        colors = []
        for pts in torch.tensor(vertices,dtype=torch.float).cuda().split(50000):
            with torch.enable_grad():
                normals = sdf_network.gradient(pts).squeeze()
                normals = F.normalize(normals, p=2, dim=-1)
            ndc_coords = torch.tensor([[0, 0, -1.0]]).cuda()  # 图像中心在NDC坐标中的位置
            nerf_cameras = self.nerfmodel.training_cameras
            p3d_camera = nerf_cameras.p3d_cameras[3]
            world_coords = p3d_camera.unproject_points(ndc_coords, world_coordinates=True)
            camera_center = p3d_camera.get_camera_center()
            ray_directions = world_coords - camera_center
            ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
            normals = normals * torch.sign((normals * ray_directions).sum(dim=-1, keepdim=True))
            normals = torch.clamp((normals + 1) / 2, 0, 1.)
            colors.append(normals.detach().cpu().numpy() * 255)
        colors = np.concatenate(colors, 0)
    else:
        colors = None

    if world_space:
        vertices = vertices * dataset.shape_scale.cpu().numpy() + dataset.shape_center.cpu().numpy()
        if crop_border:
            vertex_mask = np.all((vertices >= dataset.block_min - dataset.split_size/40) &
                                 (vertices <= dataset.block_max + dataset.split_size/40),axis=1)
            faces_mask = np.all(vertex_mask[triangles], axis=1)
            new_faces = triangles[faces_mask]
            new_indices = np.zeros(len(vertex_mask), dtype=int)
            new_indices[vertex_mask] = np.arange(np.sum(vertex_mask))
            triangles = new_indices[new_faces]
            vertices = vertices[vertex_mask]
            if vertex_color:
                colors = colors[vertex_mask]

    debug_path = os.path.join(checkpoint_path, 'meshes')
    os.makedirs(debug_path, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)
    mesh.export(os.path.join(debug_path, 'mcubes_{}_{}.ply'.format(iteration, part)))

    # shrink sdf faces
    if move_surf:
        moved_pts = []
        for pts in torch.from_numpy(np.array(vertices,dtype=np.float32)).split(200000):
            pts = pts.cuda()
            pts = (pts - dataset.shape_center) / dataset.shape_scale
            grad = sdf_network.gradient(pts).detach()
            grad_norm = F.normalize(grad, p=2, dim=-1).squeeze(1)
            pts = pts - 0.002 * grad_norm
            moved_pts.append(pts.detach())
        moved_pts = torch.cat(moved_pts, 0)
        moved_pts = moved_pts * dataset.shape_scale + dataset.shape_center
        moved_mesh = trimesh.Trimesh(moved_pts.cpu().numpy(), triangles)
        moved_mesh.export(os.path.join(sugar_checkpoint_path, 'meshes', 'mcubes_moved_{}_{}.ply'.format(iteration, part)))

        mesh = moved_mesh

    print('Marching Cubes OK.')
    return mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extract and clean mesh.')
    parser.add_argument('-s', '--scene_path',
                        type=str,
                        default='data/DTU/scan24',    # TODO: change here
                        help='(Required) path to the scene data to use.')
    parser.add_argument('-g', '--gs_checkpoint_path', type=str,
                        default='gaussian_splatting/output/DTU/scan24/',    # TODO: change here
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-o', '--output_path', type=str,
                        default='output/DTU/scan24',       # TODO: change here. DO NOT ADD '/' at end
                        help='output directory(do not include experiment name)')
    parser.add_argument('-d', '--dataset_name', default="real360", help='blender, real360, relight3d')
    args = parser.parse_args()

    output_dir = args.output_path
    scene_name = output_dir.split('/')[-1]
    if scene_name in ['Barn', 'Meetingroom', 'Courthouse', 'room']:
        part_num = 4
    else:
        part_num = 1
    sugar_checkpoint_path = output_dir
    start_iteration = 15000
    type = 'sdf'
    move_surf = type=='sdf'
    neus = Runner(sugar_checkpoint_path, None, part_num=part_num)
    neus.load_checkpoint(sugar_checkpoint_path, 'ckpt_{:06d}.pth'.format(start_iteration))
    neus.reset_datasets(sugar_checkpoint_path, None, iteration=start_iteration-1000, scene_name=scene_name)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


    vertices_list = []
    triangles_list = []
    resolution = 600

    for part in range(1, part_num + 1):
        if type == 'udf':
            evaluated_mesh = marching_cubes_udf_part(neus, start_iteration, sugar_checkpoint_path, resolution=resolution, vertex_color=False, part=part, thres=1.0)
        elif type == 'sdf':
            evaluated_mesh = marching_cubes_sdf_part(
                neus, start_iteration, sugar_checkpoint_path, resolution=resolution, vertex_color=False, part=part, thres=0.002, move_surf=move_surf)
        vertices, triangles = evaluated_mesh.vertices, evaluated_mesh.faces
        vertices_list.append(vertices)
        triangles_list.append(triangles)
        torch.cuda.empty_cache()
    all_vertices = np.vstack(vertices_list)
    all_triangles = np.vstack([
        tri + sum(map(len, vertices_list[:i]))
        for i, tri in enumerate(triangles_list)
    ])
    combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_triangles)
    combined_mesh.export(os.path.join(sugar_checkpoint_path, 'meshes', 'mesh'+type+'_merge_{}.ply'.format(start_iteration)))
    torch.cuda.empty_cache()

    # TODO: clean faces according to view visibility
    # cleaned_mesh = clean_mesh(combined_mesh, args.scene_path, args.gs_checkpoint_path, args.dataset_name)
    # cleaned_mesh.export(os.path.join(sugar_checkpoint_path, 'meshes', 'mesh'+type+'_merge_clean_{}.ply'.format(start_iteration)))
    # combined_mesh = cleaned_mesh

    # upsample points for TNT scenes for evaluation
    if scene_name in ['Ignatius', 'Truck', 'Caterpillar', 'Meetingroom', 'Barn', 'Courthouse']:
        sample_points_num = {'Truck': 8_000_000, 'Courthouse': 50_000_000, 'Meetingroom': 40_000_000, 'Ignatius': 5_000_000, 'Barn': 10_000_000, 'Caterpillar': 7_000_000}
        pts = combined_mesh.sample(sample_points_num[scene_name])
        trimesh.Trimesh(pts).export(os.path.join(sugar_checkpoint_path, 'meshes', 'mesh'+type+'_merge_{}_upsample.ply'.format(start_iteration)))
        print('up sample points for evaluation.')