#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import os
import sys
import imageio
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import copy
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def load_img(path):
    if not "." in os.path.basename(path):
        files = glob.glob(path + '.*')
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        path = files[0]
    if path.endswith(".exr"):
        if pyexr is not None:
            exr_file = pyexr.open(path)
            # print(exr_file.channels)
            all_data = exr_file.get()
            img = all_data[..., 0:3]
            if "A" in exr_file.channels:
                mask = np.clip(all_data[..., 3:4], 0, 1)
                img = img * mask
        else:
            img = imageio.imread(path)
            import pdb;
            pdb.set_trace()
        img = np.nan_to_num(img)
        hdr = True
    else:  # LDR image
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb_np(img[..., 0:3])
        hdr = False
    return img, hdr

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = copy.deepcopy(Image.open(image_path))

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' not in vertices:
        shs = np.random.random((positions.shape[0], 3)) / 255.0
        colors = SH2RGB(shs)
        normals = np.zeros((positions.shape[0], 3))
    else:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if positions.shape[0] > 2000000:
        downsample_idx = np.random.choice(positions.shape[0], 2000000, replace=False)
        positions = positions[downsample_idx]
        colors = colors[downsample_idx]
        normals = normals[downsample_idx]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if os.path.exists(os.path.join(path,'dense','fused.ply')):
        ply_path = os.path.join(path, 'dense', 'fused.ply')
    if not os.path.exists(ply_path) and not os.path.exists(bin_path) and  not os.path.exists(txt_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 500_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    elif not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = copy.deepcopy(Image.open(image_path))

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def loadCamsFromScene(path, valid_list, white_background, debug):
    with open(f'{path}/sfm_scene.json') as f:
        sfm_scene = json.load(f)

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    bbox_transform = bbox_transform.copy()
    bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    bbox_inv = np.linalg.inv(bbox_transform)

    # meta info
    image_list = sfm_scene['image_path']['file_paths']

    # camera parameters
    train_cam_infos = []
    test_cam_infos = []
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1

            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            c2w = np.linalg.inv(extrinsic)
            c2w[:3, 3] = (c2w[:4, 3] @ bbox_inv.T)[:3]
            extrinsic = np.linalg.inv(c2w)

            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]

            focal_length_x = camera_info['camera']['intrinsic']['focal'][0]
            focal_length_y = camera_info['camera']['intrinsic']['focal'][1]
            ppx = camera_info['camera']['intrinsic']['ppt'][0]
            ppy = camera_info['camera']['intrinsic']['ppt'][1]

            image_path = os.path.join(path, image_list[index])
            image_name = Path(image_path).stem

            image, is_hdr = load_img(image_path)

            depth_path = os.path.join(path + "/depths/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".tiff"))

            # if os.path.exists(depth_path):
            #     depth = load_depth(depth_path)
            #     depth *= bbox_inv[0, 0]
            # else:
            #     print("No depth map for test view.")
            #     depth = None
            depth = None

            # normal_path = os.path.join(path + "/normals/", os.path.basename(
            #     image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".pfm"))
            # if os.path.exists(normal_path):
            #     normal = load_pfm(normal_path)
            # else:
            #     print("No normal map for test view.")
            #     normal = None
            normal = None

            mask_path = os.path.join(path + "/pmasks/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".png"))
            if os.path.exists(mask_path):
                img_mask = (imageio.imread(mask_path, pilmode='L') > 0.1).astype(np.float32)
                # if pmask is available, mask the image for PSNR
                image *= img_mask[..., np.newaxis]
            else:
                img_mask = np.ones_like(image[:, :, 0])
            img_mask = np.ones_like(image[:, :, 0])

            fovx = focal2fov(focal_length_x, image.shape[1])
            fovy = focal2fov(focal_length_y, image.shape[0])
            image = Image.fromarray(np.array(image * 255, dtype=np.byte), "RGB")
            if int(index) in valid_list:
                # image *= img_mask[..., np.newaxis]
                # test_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                #                                  fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                #                                  image_path=image_path, image_name=image_name,
                #                                  depth=depth, image_mask=img_mask, normal=normal,
                #                                  width=image.shape[1], height=image.shape[0], hdr=is_hdr))
                test_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                                                 image_path=image_path, image_name=image_name,
                                                 width=image.size[0], height=image.size[1]))
            else:
                # image *= img_mask[..., np.newaxis]
                # depth *= img_mask
                # normal *= img_mask[..., np.newaxis]

                # train_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                #                                   fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                #                                   image_path=image_path, image_name=image_name,
                #                                   depth=depth, image_mask=img_mask, normal=normal,
                #                                   width=image.shape[1], height=image.shape[0], hdr=is_hdr))
                train_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                                                 image_path=image_path, image_name=image_name,
                                                 width=image.size[0], height=image.size[1]))
        if debug and i >= 5:
            break

    return train_cam_infos, test_cam_infos, bbox_transform


def readNeILFInfo(path, white_background, eval, debug=False):
    validation_indexes = []
    if eval:
        if "data_dtu" in path:
            validation_indexes = [2, 12, 17, 30, 34]
        else:
            raise NotImplementedError

    print("Reading Training transforms")
    if eval:
        print("Reading Test transforms")

    train_cam_infos, test_cam_infos, bbx_trans = loadCamsFromScene(
        f'{path}/inputs', validation_indexes, white_background, debug)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = f'{path}/inputs/model/sparse_bbx_scale.ply'
    if not os.path.exists(ply_path):
        org_ply_path = f'{path}/inputs/model/sparse.ply'

        # scale sparse.ply
        pcd = fetchPly(org_ply_path)
        inv_scale_mat = np.linalg.inv(bbx_trans)  # [4, 4]
        points = pcd.points
        xyz = (np.concatenate([points, np.ones_like(points[:, :1])], axis=-1) @ inv_scale_mat.T)[:, :3]
        normals = pcd.normals
        colors = pcd.colors

        storePly(ply_path, xyz, colors * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "NeILF": readNeILFInfo,
}