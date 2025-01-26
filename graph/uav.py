import copy
import sys
import numpy as np



sys.path.extend(['../'])
from graph import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13),
                    (13, 11), (16, 14), (14, 12), (11, 5),
                    (12, 6), (11, 12), (5, 6), (5, 0), (6, 0),
                    (1, 0), (2, 0), (3, 1), (4, 2)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

edge = inward + outward + self_link

#                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16
partition_label   = [3, 3, 3, 3, 3, 0, 0, 1, 2, 1, 2, 0, 0, 4, 5, 4, 5]

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


# 将人体分为6个部分：左腿、右腿、躯干、左手、右手、头部
class MyGraph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.edge = edge

        self.hop_dis = tools.get_hop_distance(
            self.num_node, self.edge, max_hop=1)
        self.A = self.get_adjacency_matrix_A_partly(partition_label, labeling_mode)
        self.partition_label = partition_label

    def get_adjacency_matrix_A_partly(self, partition_label, labeling_mode=None):
        if labeling_mode is None:
            return self.A6
        if labeling_mode == 'spatial':
            A = np.zeros((17, 17), dtype=np.int32)

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

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
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