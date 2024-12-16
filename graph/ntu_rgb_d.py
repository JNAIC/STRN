import copy
import sys
import numpy as np


sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

edge = inward + outward + self_link

#                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
partition_label   = [3, 3, 2, 2, 3, 1, 1, 1, 3, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 3, 1, 1, 0, 0]
partition_label_2 = [3, 3, 2, 2, 3, 3, 1, 1, 3, 3, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 3, 1, 1, 0, 0]
partition_label_3 = [3, 3, 2, 2, 3, 3, 1, 1, 3, 3, 0, 0, 3, 5, 5, 5, 3, 4, 4, 4, 3, 1, 1, 0, 0]
partition_label_4 = [3, 3, 2, 2, 3, 7, 7, 1, 3, 6, 6, 0, 5, 5, 5, 5, 4, 4, 4, 4, 3, 1, 1, 0, 0]


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.edge = edge

        self.hop_dis = tools.get_hop_distance(
            self.num_node, self.edge, max_hop=1)
        self.A6 = self.get_adjacency_matrix_A_k(6, tools.get_k_body_parts_ntu(6), labeling_mode)
        self.A = self.get_adjacency_matrix_A_partly(partition_label, labeling_mode)
        self.A8 = self.get_adjacency_matrix_A_k(8, tools.get_k_body_parts_ntu(8), labeling_mode)
        self.A3 = self.get_adjacency_matrix_A3(labeling_mode)
        self.partition_label = partition_label
        self.spd_A = copy.deepcopy(self.A6)

    def get_adjacency_matrix_A_k(self, k, partition_body, labeling_mode=None, ones=False):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            adjacency_matrix = tools.get_spatial_graph_new(num_node, edge)
            if k == 6:
                Ak = np.zeros((6, 25, 25), dtype=np.float32)
            elif k == 8:
                Ak = np.zeros((8, 25, 25), dtype=np.float32)
            else:
                raise ValueError()

            if not ones:
                for hop in range(2):
                    for i in range(self.num_node):
                        for j in range(self.num_node):
                            if self.hop_dis[j, i] == hop:
                                part_indices_j = tools.get_part_index(partition_body, j)
                                Ak[part_indices_j, i, j] = adjacency_matrix[i, j]
            else:
                for hop in range(2):
                    for i in range(self.num_node):
                        for j in range(self.num_node):
                            if self.hop_dis[j, i] == hop:
                                part_indices_j = tools.get_part_index(partition_body, j)
                                Ak[part_indices_j, i, j] = 1.0
        else:
            raise ValueError()
        return Ak

    def get_adjacency_matrix_A_partly(self, partition_label, labeling_mode=None):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            A = np.zeros((25, 25), dtype=np.int32)

            h = {}
            cnt = max(partition_label) + 1
            for i in range(self.num_node):
                for j in range(self.num_node):
                    indices_i, indices_j = partition_label[i], partition_label[j]
                    if self.hop_dis[j, i] <= 1:
                        if indices_i == indices_j:
                            A[i, j] = A[j, i] = indices_j
                        else:
                            A[i, j] = indices_i
                            A[j, i] = indices_j
                    else:
                        if not h.get(f'{indices_i}-{indices_j}'):
                            h[f'{indices_i}-{indices_j}'] = cnt
                            cnt = cnt + 1
                        A[i, j] = h[f'{indices_i}-{indices_j}']

                        if not h.get(f'{indices_j}-{indices_i}'):
                            h[f'{indices_j}-{indices_i}'] = cnt
                            cnt = cnt + 1
                        A[j, i] = h[f'{indices_j}-{indices_i}']


        else:
            raise ValueError()
        return A


    def get_adjacency_matrix_A3(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A3
        if labeling_mode == 'spatial':
            A3 = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A3


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    graph = MyGraph()
    A = graph.A + 1


    plt.figure(figsize=(10, 10))
    # 绘制矩阵 A，使用 jet 颜色映射
    img = plt.imshow(A, cmap='jet', interpolation='nearest')
    # 在每个像素上方显示相应的数值
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # 检查像素的颜色深浅，决定显示的文字颜色
            if np.mean(img.cmap(img.norm(A[i, j]))) > 0.6:
                plt.text(j, i, str(A[i, j]), ha='center', va='center', color='black')
            else:
                plt.text(j, i, str(A[i, j]), ha='center', va='center', color='white')
    # 设置标题
    plt.title('Matrix A')
    # 显示图像
    plt.show()