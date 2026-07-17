import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools
from graph.infogcn import tools as tools_infogcn

# CARE-PD skeletons use the 22-joint SMPL/HumanML3D order:
# 1 pelvis, 2 left_hip, 3 right_hip, 4 spine1, 5 left_knee,
# 6 right_knee, 7 spine2, 8 left_ankle, 9 right_ankle,
# 10 spine3, 11 left_foot, 12 right_foot, 13 neck,
# 14 left_collar, 15 right_collar, 16 head, 17 left_shoulder,
# 18 right_shoulder, 19 left_elbow, 20 right_elbow,
# 21 left_wrist, 22 right_wrist.
num_node = 22
self_link = [(i, i) for i in range(num_node)]

# ST-GCN spatial graph: directed from distal joints toward the torso center.
inward_ori_index = [
    # left leg -> pelvis
    (11, 8), (8, 5), (5, 2), (2, 1),
    # right leg -> pelvis
    (12, 9), (9, 6), (6, 3), (3, 1),
    # pelvis -> upper torso
    (1, 4), (4, 7), (7, 10),
    # head/neck -> upper torso
    (16, 13), (13, 10),
    # left arm -> upper torso
    (21, 19), (19, 17), (17, 14), (14, 10),
    # right arm -> upper torso
    (22, 20), (20, 18), (18, 15), (15, 10),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# InfoGCN-style tree: directed from upper torso/root toward distal joints.
inward_ori_index_infogcn = [
    # torso and head
    (10, 7), (7, 4), (4, 1), (10, 13), (13, 16),
    # left leg
    (1, 2), (2, 5), (5, 8), (8, 11),
    # right leg
    (1, 3), (3, 6), (6, 9), (9, 12),
    # left arm
    (10, 14), (14, 17), (17, 19), (19, 21),
    # right arm
    (10, 15), (15, 18), (18, 20), (20, 22),
]
inward_infogcn = [(i - 1, j - 1) for (i, j) in inward_ori_index_infogcn]
outward_infogcn = [(j, i) for (i, j) in inward_infogcn]
neighbor_infogcn = inward_infogcn + outward_infogcn


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.outward_infogcn = outward_infogcn
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools_infogcn.get_adjacency_matrix(self.outward_infogcn, self.num_node)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
