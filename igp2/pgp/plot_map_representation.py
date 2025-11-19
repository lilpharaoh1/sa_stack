import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrow
import numpy as np

MAP_FADE = 0.3

def world_to_ego_batch(xs, ys, agent_pose):
    cx, cy, heading = agent_pose
    dx = np.array(xs) - cx
    dy = np.array(ys) - cy
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    x_ego = cos_h * dx - sin_h * dy
    y_ego = sin_h * dx + cos_h * dy
    return x_ego, y_ego

def ego_to_world_batch(xs, ys, agent_pose):
    cx, cy, heading = agent_pose
    
    # Rotation by +heading (ego → world)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    # Apply rotation
    x_rot = cos_h * np.array(xs) - sin_h * np.array(ys)
    y_rot = sin_h * np.array(xs) + cos_h * np.array(ys)
    
    # Apply translation
    x_world = x_rot + cx
    y_world = y_rot + cy
    
    return x_world, y_world

def heading_from_history(agent_history):
    return np.arctan2(agent_history[-1][1] - agent_history[-2][1], agent_history[-1][0] - agent_history[-2][0])
    
def plot_map_representation(map_representation, nodes, edges, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    colors = plt.get_cmap("tab10").colors

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_facecolor("grey")
    ax.set_aspect("equal")

    # --- Plot Lane Graph (already ego-frame) ---
    node_ids = list(nodes.keys())
    idx_to_id = {idx: seg_id for idx, seg_id in enumerate(node_ids)}
    id_to_idx = {seg_id: idx for idx, seg_id in enumerate(node_ids)}
    N = len(node_ids)

    for node_id, node in nodes.items():
        x, y = node["pose"][:2]
        ax.scatter(x, y, color="r", marker="o", linewidth=5)
        ax.text(x, y, id_to_idx[node_id], fontsize=13)

    for start_id, end_id in edges:
        if not start_id in id_to_idx or not start_id in id_to_idx:
            continue
        s = nodes[start_id]["pose"]
        e = nodes[end_id]["pose"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        ax.arrow(s[0], s[1], dx, dy, color='b', head_width=1, width=0.5, length_includes_head=True) 

    for edge_idx, (edge, edge_type) in enumerate(zip(map_representation['s_next'], map_representation['edge_type'])):
        if edge[-1] == 0 and edge_type[-1] == 0:
            continue
        for e, e_t in zip(edge[:-1], edge_type[:-1]):
            if e == 0.0:
                continue
            start_feat = map_representation["lane_node_feats"][edge_idx][0]
            end_feat = map_representation["lane_node_feats"][int(e)][0]
            s = start_feat[:2]
            e = end_feat[:2]
            dx, dy = e[0] - s[0], e[1] - s[1]
            ax.arrow(s[0], s[1], dx, dy, color='r' if e_t == 1 else 'g', head_width=0.4, width=0.2, length_includes_head=True)

    for feat_node in map_representation["lane_node_feats"]:
        if np.sum(feat_node[0]) == 0.0:
            continue
        x, y, yaw = feat_node[0][:3]

        # Arrow properties
        arrow_length = 0.5
        arrow_color = 'orange'
        arrow_width = 0.4

        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        # ax.scatter(x, y, color="r", marker="o", linewidth=3)
        ax.arrow(x, y, dx, dy, head_width=arrow_width, head_length=arrow_width * 1.5,
                fc=arrow_color, ec=arrow_color, length_includes_head=True)

        # # Convert heading angles to directional vectors (u, v)
        # u = np.cos(yaw)  # x-component of arrow
        # v = np.sin(yaw)  # y-component of arrow
        # plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=0.01, color='blue')
        # ax.scatter(x, y, color="r", marker="o", linewidth=3)
        
    return ax

    


if __name__ == '__main__':
    pass
    # xodr_file = f"scenarios/maps/scenario1.xodr"
    # odr_map = Map.parse_from_opendrive(xodr_file)
    # dataset = Dataset.parse_from_opendrive(xodr_file)
    # # dataset.generate_graph(agent_pose=[90, -75, np.pi / 2])
    # dataset.generate_graph(agent_pose=[35.0, -1.8, np.pi / 2])
    # # dataset.generate_graph(agent_pose=[64, 0, np.pi/2])
    # # plot_map(odr_map, markings=True, midline=True)
    # plot_vector_map(dataset, odr_map, markings=True, agent=True)

    # plt.show()

    # print(dataset.get_map_representation()["s_next"][0:6])