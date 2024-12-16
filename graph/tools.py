import numpy as np


def get_k_body_parts_ntu(k):
    if k == 6:
        left_arm = [10, 11, 12, 24, 25]
        right_arm = [6, 7, 8, 22, 23]
        head = [3, 4]
        body = [1, 2, 5, 9, 21]  # 1 2 5 6 9 10 21
        left_leg = [17, 18, 19, 20]
        right_leg = [13, 14, 15, 16]

        partition_body = [left_arm, right_arm, head, body, left_leg, right_leg]
        partition_body = [[index - 1 for index in part] for part in partition_body]
    elif k == 8:
        left_arm = [10, 11]
        right_arm = [6, 7]
        left_hand = [12, 24, 25]
        right_hand = [8, 22, 23]
        head = [3, 4]
        body = [1, 2, 5, 9, 21]
        left_leg = [17, 18, 19, 20]
        right_leg = [13, 14, 15, 16]

        partition_body = [left_arm, right_arm, left_hand, right_hand, head, body, left_leg, right_leg]
        partition_body = [[index - 1 for index in part] for part in partition_body]

    return partition_body


def get_k_body_parts_ucla(k):
    if k == 6:
        left_arm = [5, 6, 7, 8]
        right_arm = [9, 10, 11, 12]
        head = [4]
        body = [1, 2, 3]
        left_leg = [13, 14, 15, 16]
        right_leg = [17, 18, 19, 20]

        partition_body = [left_arm, right_arm, head, body, left_leg, right_leg]
        partition_body = [[index - 1 for index in part] for part in partition_body]
    elif k == 8:
        left_arm = [5, 6]
        right_arm = [9, 10]
        left_hand = [7, 8]
        right_hand = [11, 12]
        head = [4]
        body = [1, 2, 3]
        left_leg = [13, 14, 15, 16]
        right_leg = [17, 18, 19, 20]

        partition_body = [left_arm, right_arm, left_hand, right_hand, head, body, left_leg, right_leg]
        partition_body = [[index - 1 for index in part] for part in partition_body]

    return partition_body


def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    # print(In)
    return A


def get_spatial_graph_new(num_node, edge):
    A = normalize_digraph(edge2mat(edge, num_node))
    return A


def get_spatial_graph_new_new(num_node, left_arm_part, right_arm_part,
                              head_part, body_part, left_leg_part, right_leg_part):
    LA = normalize_digraph(edge2mat(left_arm_part, num_node))
    RA = normalize_digraph(edge2mat(right_arm_part, num_node))
    H = normalize_digraph(edge2mat(head_part, num_node))
    B = normalize_digraph(edge2mat(body_part, num_node))
    LL = normalize_digraph(edge2mat(left_leg_part, num_node))
    RL = normalize_digraph(edge2mat(right_leg_part, num_node))
    A = np.stack((LA, RA, H, B, LL, RL))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_part_index(partition_body, v):
    for idx, part in enumerate(partition_body):
        if v in part:
            return idx
