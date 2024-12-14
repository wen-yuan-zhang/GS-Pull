import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
from pytorch3d.ops import knn_points
from tqdm import tqdm


def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1)
    point_gt = point_gt.unsqueeze(0)

    distances = torch.sqrt(torch.sum((point_batch - point_gt) ** 2, axis=-1) + 1e-12)
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx


def process_data(data_dir, dataname):
    if os.path.exists(os.path.join(data_dir, 'input', dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, 'input', dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.xyz'):
        pointcloud = np.loadtxt(os.path.join(data_dir, 'input', dataname) + '.xyz')
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.npy'):
        pointcloud = np.load(os.path.join(data_dir, 'input', dataname) + '.npy')
    else:
        print('Only support .ply, .xyz or .npy data. Please adjust your data format.')
        exit()
    shape_scale = np.max(
        [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
         np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
    shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                    (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                    (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000 // POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[point_idx, :]
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, 100, axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])

    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []

    # 小范围采样
    for i in tqdm(range(QUERY_EACH)):
        scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1, POINT_NUM, 3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            # nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            # nearest_points = pointcloud[nearest_idx]
            knns = knn_points(torch.tensor(tt[j]).float().cuda()[None], torch.tensor(pointcloud).float().cuda()[None],
                              K=1)
            nearest_idx = knns.idx[0][:, 0].cpu().numpy()
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1, 3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1, 3)
        sample_near.append(sample_near_tmp)
    # 大范围采样
    # for i in tqdm(range(QUERY_EACH)):
    #     scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    #     scale *= 5
    #     tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
    #     sample.append(tt)
    #     tt = tt.reshape(-1,POINT_NUM,3)
    #
    #     sample_near_tmp = []
    #     for j in range(tt.shape[0]):
    #         # nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
    #         # nearest_points = pointcloud[nearest_idx]
    #         knns = knn_points(torch.tensor(tt[j]).float().cuda()[None], torch.tensor(pointcloud).float().cuda()[None],K=1)
    #         nearest_idx = knns.idx[0][:, 0].cpu().numpy()
    #         nearest_points = pointcloud[nearest_idx]
    #         nearest_points = np.asarray(nearest_points).reshape(-1,3)
    #         sample_near_tmp.append(nearest_points)
    #     sample_near_tmp = np.asarray(sample_near_tmp)
    #     sample_near_tmp = sample_near_tmp.reshape(-1,3)
    #     sample_near.append(sample_near_tmp)

    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)

    os.makedirs(os.path.join(data_dir, 'query_data'), exist_ok=True)
    np.savez(os.path.join(data_dir, 'query_data', dataname) + '.npz', sample=sample, point=pointcloud,
             sample_near=sample_near)
    return shape_center, shape_scale


class Dataset:
    def __init__(self, pointcloud, part=1, scene_name='Barn', old_pc_bbx=None, old_shape_center=None, old_shape_scale=None):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')

        ### process data
        self.rescale = 1
        if old_pc_bbx is None:
            pc_bbx = np.array([[np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0])],
                               [np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1])],
                               [np.min(pointcloud[:, 2]), np.max(pointcloud[:, 2])]])
            print('Use crop box to reconstruct foreground area.')
            if scene_name in ['Truck']:
                pc_bbx = np.array([[-2.5181, 3.8502], [-0.4665, 1.6685], [-2.2749, 2.6918]])
            elif scene_name in ['Meetingroom']:
                pc_bbx = np.array([[-6.0672, 6.4682], [-3.4091, 1.8407], [-7.5986, 6.5010]])
            elif scene_name in ['Courthouse']:
                pc_bbx = np.array([[-4.6782, 2.0279], [-3.1852, 1.2774], [-1.6823, 4.8831]])
            elif 'scan' in scene_name:      # DTU
                pc_bbx = np.array([[-1.0827, 0.8112], [-1.2027, 0.8358], [-0.8887, 0.8409]])
            elif scene_name in ['bicycle']:
                pc_bbx = np.array([[-0.9149, 2.1577], [0.1237, 2.1134], [-1.5676, 2.6775]])
            elif scene_name in ['bonsai']:
                pc_bbx = np.array([[-1.0871, 1.7355], [0.5233, 4.9407], [0.5918, 4.5681]])
            elif scene_name in ['room']:
                pc_bbx = np.array([[-6.0511, 6.9763], [-4.7568,8.8471], [-12.3005,6.8691]])
            elif scene_name in ['stump']:
                pc_bbx = np.array([[-2.7293,3.5282], [-0.5877,3.8539], [0.0733,4.0487]])
            elif scene_name in ['counter']:
                pc_bbx = np.array([[-2.6283,4.3568], [0.1092,4.5699], [-1.9184,4.3053]])
            elif scene_name in ['kitchen']:
                pc_bbx = np.array([[-2.5960,2.3708], [0.4092,3.4123], [-0.3833,3.3857]])
            elif scene_name in ['garden']:
                pc_bbx = np.array([[-2.3880,2.6762], [0.3397,3.7876], [-0.3535,3.6491]])
            elif scene_name in ['treehill']:
                pc_bbx = np.array([[-2.0203,2.0408], [-2.5654,2.0888], [-1.8776,2.4737]])
            elif scene_name in ['flowers']:
                pc_bbx = np.array([[-2.0380,2.3308], [-0.8755,1.5160], [-2.3950,1.8431]])
        else:
            pc_bbx = old_pc_bbx

        # split 4 blocks for very large scenes
        if scene_name in ['Barn', 'Meetingroom', 'Courthouse', 'room']:
            split_size = (pc_bbx[:,1] - pc_bbx[:,0]) / 2
            split_size[1] = 0       # along y axis

            if part == 1:
                block_min = np.array([pc_bbx[0,0],pc_bbx[1,0],pc_bbx[2,0]])
                block_max = np.array([pc_bbx[0,0]+split_size[0],pc_bbx[1,1]+split_size[1],pc_bbx[2,0]+split_size[2]])
                part_select_idx = np.all((pointcloud > block_min - split_size/20) & (pointcloud < block_max + split_size/20), axis=1)
            elif part == 2:
                block_min = np.array([pc_bbx[0,0]+split_size[0],pc_bbx[1,0],pc_bbx[2,0]])
                block_max = np.array([pc_bbx[0,1],pc_bbx[1,1]+split_size[1],pc_bbx[2,0]+split_size[2]])
                part_select_idx = np.all(
                    (pointcloud > block_min - split_size / 20) & (pointcloud < block_max + split_size / 20), axis=1)
            elif part == 3:
                block_min = np.array([pc_bbx[0,0]+split_size[0],pc_bbx[1,0],pc_bbx[2,0]+split_size[2]])
                block_max = np.array([pc_bbx[0,1],pc_bbx[1,1]+split_size[1],pc_bbx[2,1]])
                part_select_idx = np.all(
                    (pointcloud > block_min - split_size / 20) & (pointcloud < block_max + split_size / 20), axis=1)

            elif part == 4:
                block_min = np.array([pc_bbx[0,0],pc_bbx[1,0],pc_bbx[2,0]+split_size[2]])
                block_max = np.array([pc_bbx[0,0]+split_size[0],pc_bbx[1,1]+split_size[1],pc_bbx[2,1]])
                part_select_idx = np.all(
                    (pointcloud > block_min - split_size / 20) & (pointcloud < block_max + split_size / 20), axis=1)
        else:
            block_min = pc_bbx[:,0]
            block_max = pc_bbx[:,1]
            split_size = np.array([0,0,0])
            part_select_idx = np.all(
                (pointcloud > block_min - split_size / 20) & (pointcloud < block_max + split_size / 20), axis=1)
        self.part_select_idx = torch.tensor(part_select_idx, dtype=torch.bool).to(self.device)
        self.block_min, self.block_max, self.split_size = block_min, block_max, split_size
        pointcloud = pointcloud[part_select_idx]
        shape_scale = np.max(
            [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
             np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
        shape_scale /= self.rescale
        shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                        (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                        (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
        if old_shape_center is None:
            self.shape_center = torch.tensor(shape_center,dtype=torch.float).to(self.device)
        else:
            self.shape_center = old_shape_center
            print('load old shape center.')
        if old_shape_scale is None:
            self.shape_scale = torch.tensor([shape_scale],dtype=torch.float).to(self.device)
        else:
            self.shape_scale = old_shape_scale
            print('load old shape center.')
        pointcloud = pointcloud - self.shape_center.detach().cpu().numpy()
        pointcloud = pointcloud / self.shape_scale.detach().cpu().numpy()

        POINT_NUM = pointcloud.shape[0] // 60
        POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
        QUERY_EACH = 1000000 // POINT_NUM_GT
        if QUERY_EACH < 10:
            QUERY_EACH = 10
        self.query_each = QUERY_EACH

        point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
        self.downsample_idx = torch.tensor(point_idx, dtype=torch.long).to(self.device)
        pointcloud = pointcloud[point_idx, :]
        ptree = cKDTree(pointcloud)
        sigmas = []
        for p in np.array_split(pointcloud, 100, axis=0):
            d = ptree.query(p, 51)
            sigmas.append(d[0][:, -1])

        self.pc_bbx = pc_bbx
        self.object_bbox_min = pc_bbx[:,0] - 0.05
        self.object_bbox_max = pc_bbx[:,1] + 0.05
        self.object_bbox_min = np.array([-0.6, -0.6, -0.6])
        self.object_bbox_max = np.array([0.6, 0.6, 0.6])
        print('bd:', self.object_bbox_min, self.object_bbox_max)

        sigmas = np.concatenate(sigmas)
        sample = []
        sample_near = []
        sample_near_idx = []
        # near sampling
        for i in tqdm(range(QUERY_EACH)):
            thres = 0.25
            scale = thres if thres * np.sqrt(POINT_NUM_GT / 20000) < thres else thres * np.sqrt(POINT_NUM_GT / 20000)
            tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
            sample.append(tt)
            tt = tt.reshape(-1, POINT_NUM, 3)
            sample_near_tmp = []
            sample_near_idx_tmp = []
            for j in range(tt.shape[0]):
                knns = knn_points(torch.tensor(tt[j]).float().cuda()[None],
                                  torch.tensor(pointcloud).float().cuda()[None], K=1)
                nearest_idx = knns.idx[0][:, 0].cpu().numpy()
                sample_near_idx_tmp.append(nearest_idx)
                nearest_points = pointcloud[nearest_idx]
                nearest_points = np.asarray(nearest_points).reshape(-1, 3)
                sample_near_tmp.append(nearest_points)
            sample_near_tmp = np.asarray(sample_near_tmp)
            sample_near_tmp = sample_near_tmp.reshape(-1, 3)
            sample_near.append(sample_near_tmp)
            sample_near_idx_tmp = np.asarray(sample_near_idx_tmp).reshape(-1)
            sample_near_idx.append(sample_near_idx_tmp)
        # all space sampling
        if scene_name in ['treehill', 'Meetingroom']:
            all_space_sample_num = {'Meetingroom': 3, 'Barn': 1, 'treehill': 2}
            for i in tqdm(range(all_space_sample_num[scene_name])):
                space_samples = np.random.rand(*pointcloud.shape)
                space_samples *= (self.object_bbox_max - self.object_bbox_min)
                space_samples += self.object_bbox_min
                sample.append(space_samples)
                knns = knn_points(torch.tensor(space_samples).float().cuda()[None],
                                  torch.tensor(pointcloud).float().cuda()[None], K=1)
                nearest_idx = knns.idx[0][:, 0].cpu().numpy()
                nearest_points = pointcloud[nearest_idx]
                nearest_points = np.asarray(nearest_points).reshape(-1, 3)
                sample_near.append(nearest_points)
                sample_near_idx.append(nearest_idx)
        # far sampling
        if scene_name in ['bicycle', 'room', 'Meetingroom', 'stump', 'counter', 'bonsai', 'garden', 'treehill', 'flowers', 'Caterpillar'] or 'scan' in scene_name:  # TODO: add Barn far samp
            for i in tqdm(range(1)):
                thres = 0.75
                scale = thres if thres * np.sqrt(POINT_NUM_GT / 20000) < thres else thres * np.sqrt(POINT_NUM_GT / 20000)
                tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
                sample.append(tt)
                tt = tt.reshape(-1, POINT_NUM, 3)
                sample_near_tmp = []
                sample_near_idx_tmp = []
                for j in range(tt.shape[0]):
                    knns = knn_points(torch.tensor(tt[j]).float().cuda()[None],
                                      torch.tensor(pointcloud).float().cuda()[None], K=1)
                    nearest_idx = knns.idx[0][:, 0].cpu().numpy()
                    sample_near_idx_tmp.append(nearest_idx)
                    nearest_points = pointcloud[nearest_idx]
                    nearest_points = np.asarray(nearest_points).reshape(-1, 3)
                    sample_near_tmp.append(nearest_points)
                sample_near_tmp = np.asarray(sample_near_tmp)
                sample_near_tmp = sample_near_tmp.reshape(-1, 3)
                sample_near.append(sample_near_tmp)
                sample_near_idx_tmp = np.asarray(sample_near_idx_tmp).reshape(-1)
                sample_near_idx.append(sample_near_idx_tmp)

        sample = np.asarray(sample)
        sample_near = np.asarray(sample_near)
        sample_near_idx = np.asarray(sample_near_idx, dtype=np.int64).reshape(-1)

        ### end process data

        self.point = np.asarray(sample_near).reshape(-1, 3)
        self.sample = np.asarray(sample).reshape(-1, 3)
        self.point_gt = np.asarray(pointcloud).reshape(-1, 3)
        self.sample_points_num = self.sample.shape[0] - 1

        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        self.point_idx = torch.from_numpy(sample_near_idx).to(self.device).long()

        print('NP Load data: End')

    def get_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num // 10, batch_size, replace=False)
        index = index_fine * 10 + index_coarse  # for accelerating random choice operation
        points = self.point[index]
        sample = self.sample[index]
        points_idx = self.point_idx[index]
        return points, sample, self.point_gt, points_idx

