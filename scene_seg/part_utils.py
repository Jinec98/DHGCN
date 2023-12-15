import numpy as np
from tqdm import tqdm

def write_plyfile(file_name, point_cloud):
    f = open(file_name + '.ply', 'w')
    init_str = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(point_cloud)) + \
               "\nproperty float x\nproperty float y\nproperty float z\n" \
               "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(init_str)
    for i in range(len(point_cloud)):
        f.write(str(round(float(point_cloud[i][0]), 6)) + ' ' + str(round(float(point_cloud[i][1]), 6)) + ' ' +
                str(round(float(point_cloud[i][2]), 6)) + '\n')
    f.close()


def get_bbox(point_cloud):
    """Check if two bbox are intersected
    Args:
        point_cloud (_type_): (N, C)
    """
    point_cloud = np.array(point_cloud).T  # C,N
    x_diff = max(point_cloud[0]) - min(point_cloud[0])
    y_diff = max(point_cloud[1]) - min(point_cloud[1])
    z_diff = max(point_cloud[2]) - min(point_cloud[2])
    return min(point_cloud[0]), min(point_cloud[1]), min(point_cloud[2]), x_diff, y_diff, z_diff


def is_connected(point_cloud_1, point_cloud_2, theta=0.2):
    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return False
    x_start_1, y_start_1, z_start_1, x_diff_1, y_diff_1, z_diff_1 = get_bbox(
        point_cloud_1)
    x_start_2, y_start_2, z_start_2, x_diff_2, y_diff_2, z_diff_2 = get_bbox(
        point_cloud_2)

    # scale bbox
    x_end_1 = x_start_1 + x_diff_1 + x_diff_1*theta
    y_end_1 = y_start_1 + y_diff_1 + y_diff_1*theta
    z_end_1 = z_start_1 + z_diff_1 + z_diff_1*theta
    x_end_2 = x_start_2 + x_diff_2 + x_diff_2*theta
    y_end_2 = y_start_2 + y_diff_2 + y_diff_2*theta
    z_end_2 = z_start_2 + z_diff_2 + z_diff_2*theta

    x_start_1 -= x_diff_1*theta
    y_start_1 -= y_diff_1*theta
    z_start_1 -= z_diff_1*theta
    x_start_2 -= x_diff_2*theta
    y_start_2 -= y_diff_2*theta
    z_start_2 -= y_diff_2*theta

    # check if two bbox are intersected
    if (x_start_1 <= x_start_2 <= x_end_1 or x_start_2 <= x_start_1 <= x_end_2) and (y_start_1 <= y_start_2 <= y_end_1 or y_start_2 <= y_start_1 <= y_end_2) and (z_start_1 <= z_start_2 <= z_end_1 or z_start_2 <= z_start_1 <= z_end_2):
        return True
    else:
        return False


def split_part(data, split_num=3, max_distance=3):
    # data: (B,N,C)
    B, N, C = data.shape
    print('Split point cloud to part...')
    max_p = np.max(data, axis=1)[:, np.newaxis, :]  # (B,N,C)->(B,1,C)
    min_p = np.min(data, axis=1)[:, np.newaxis, :]
    diff = max_p - min_p  # (B,C)

    diff /= split_num
    diff[diff == 0] = 1e-5
    p2v_indices = ((data - min_p) / diff).astype(int)  # (B,N,C)
    p2v_indices[p2v_indices == split_num] = split_num - 1
    p2v_indices[p2v_indices > split_num] = 1  
    # voxel index : (B,N,C) -> (B,N)
    p2v_indices = p2v_indices[:, :, 2] + split_num * p2v_indices[:, :,
                                                         1] + split_num**2 * p2v_indices[:, :, 0] 

    print('Get part hop distance...')
    print('--1 Get adjacency matrix...')
    adjacency = np.zeros((B, split_num**3, split_num ** 3)).astype(int)

    for b, pc in enumerate(tqdm(data)):
        for i in range(split_num**3):
            for j in range(i): 
                pc_i = pc[p2v_indices[b] == i]
                pc_j = pc[p2v_indices[b] == j]
                if is_connected(pc_i, pc_j):
                    adjacency[b, i, j] = 1
                    adjacency[b, j, i] = 1

    
    print('--2 Get hop distance...')
    part_distance = np.empty((B, split_num**3, split_num ** 3))
    part_distance.fill(1e3)
    part_distance[adjacency == 1] = 1
    batch_index = np.arange(B).reshape((B, 1)).repeat(split_num ** 3, axis=1)
    part_distance[batch_index, np.arange(
        split_num ** 3), np.arange(split_num ** 3)] = 0  # set self distance=0
    part_distance = part_distance.astype(int)

    for b, pc in enumerate(tqdm(data)):
        # Floyd–Warshall algorithm: O(V^3)
        for k in range(split_num ** 3):
            for i in range(split_num ** 3):
                for j in range(split_num ** 3):
                    if part_distance[b, i, j] > part_distance[b, i, k] + part_distance[b, k, j]:
                        part_distance[b, i, j] = part_distance[b,
                                                               i, k] + part_distance[b, k, j]
    part_distance[part_distance > max_distance] = max_distance
    
    
    return p2v_indices, part_distance



