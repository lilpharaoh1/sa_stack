import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrow
import numpy as np

from igp2.opendrive import Map
from igp2.opendrive.elements.road_lanes import LaneTypes

from igp2.vector_map.dataset import Dataset

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


def plot_pgp_trajectories(trajectories, agent_history, odr_map, ax: plt.Axes = None, scenario_config=None, **kwargs) -> plt.Axes:
    """Plot OpenDRIVE map, optionally transformed to the ego frame."""
    colors = plt.get_cmap("tab10").colors

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(odr_map.name)

    if odr_map is None:
        return ax

    ax.set_xlim([odr_map.west, odr_map.east])
    ax.set_ylim([odr_map.south, odr_map.north])
    ax.set_facecolor("grey")

    if kwargs.get("plot_background", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw background")
        else:
            background_path = scenario_config.data_root + '/' + scenario_config.background_image
            background = imageio.imread(background_path)
            rescale_factor = scenario_config.background_px_to_meter
            extent = (0, int(background.shape[1] * rescale_factor),
                      -int(background.shape[0] * rescale_factor), 0)
            plt.imshow(background, extent=extent)

    if kwargs.get("plot_buildings", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw buildings")
        else:
            buildings = scenario_config.buildings

            for building in buildings:
                # Add the first point also at the end, so we plot a closed contour of the obstacle.
                building.append((building[0]))
                plt.plot(*list(zip(*building)), color="black")

    if kwargs.get("plot_goals", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw buildings")
        else:
            goals = scenario_config.goals

            for goal in goals:
                plt.plot(*goal, color="r", marker='o', ms=10)

    if kwargs.get("ignore_roads", False):
        return ax

    for road_id, road in odr_map.roads.items():
        boundary = road.boundary.boundary
        if road.junction is None or not kwargs.get("hide_road_bounds_in_junction", False):
            if boundary.geom_type == "LineString":
                ax.plot(boundary.xy[0],
                        boundary.xy[1],
                        color=kwargs.get("road_color", "k"))
            elif boundary.geom_type == "MultiLineString":
                for b in boundary.geoms:
                    ax.plot(b.xy[0],
                            b.xy[1],
                            color=kwargs.get("road_color", "orange"))

        color = kwargs.get("midline_color", colors[road_id % len(colors)] if kwargs.get("road_ids", False) else "r")
        if kwargs.get("midline", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0:
                        continue
                    if not lane.type == LaneTypes.DRIVING and kwargs.get("drivable", False):
                        continue
                    if kwargs.get("midline_direction", False):
                        x = np.array(lane.midline.xy[0])
                        y = np.array(lane.midline.xy[1])
                        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                                  width=0.0025, headwidth=2,
                                  scale_units='xy', angles='xy', scale=1, color="red")
                    else:
                        ax.plot(lane.midline.xy[0],
                                lane.midline.xy[1],
                                color=color)

        if kwargs.get("road_ids", False):
            mid_point = len(road.midline.xy) // 2
            ax.text(road.midline.xy[0][mid_point],
                    road.midline.xy[1][mid_point],
                    road.id,
                    color=color, fontsize=15)

        if kwargs.get("markings", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    for marker in lane.markers:
                        line_styles = marker.type_to_linestyle
                        for i, style in enumerate(line_styles):
                            if style is None:
                                continue
                            df = 0.13  # Distance between parallel lines
                            side = "left" if lane.id <= 0 else "right"
                            line = lane.reference_line.parallel_offset(i * df, side=side)
                            ax.plot(line.xy[0], line.xy[1],
                                    color=marker.color_to_rgb,
                                    linestyle=style,
                                    linewidth=marker.plot_width)

    for junction_id, junction in odr_map.junctions.items():
        if junction.boundary.geom_type == "Polygon":
            ax.fill(junction.boundary.boundary.xy[0],
                    junction.boundary.boundary.xy[1],
                    color=kwargs.get("junction_color", (0.941, 1.0, 0.420, 0.5)))
        else:
            if hasattr(junction.boundary, "geoms"):
                geoms = junction.boundary.geoms
            else:
                geoms = junction.boundary
            for polygon in geoms:
                ax.fill(polygon.boundary.xy[0],
                        polygon.boundary.xy[1],
                        color=kwargs.get("junction_color", (0.941, 1.0, 0.420, 0.5)))

    traj_colors = plt.cm.get_cmap("tab10")  # Or any other colormap
    for i, (agent_id, agent_states) in enumerate(agent_history.items()):
        # Agent colour
        base_color = traj_colors(i % 10)  # keep within range of cmap

        # Plot agent history
        traj_history = np.array(list(agent_states))
        x_history = traj_history[:, 0]
        y_history = traj_history[:, 1]
        ax.plot(x_history, y_history, color=base_color, marker='o')

        # Plot predicted agent futures
        traj_preds = trajectories[agent_id]
        for traj_pred in traj_preds:
            y_pred = traj_pred[:, 0].cpu().detach().numpy()
            x_pred = traj_pred[:, 1].cpu().detach().numpy()
            x_pred, y_pred = ego_to_world_batch(x_pred, y_pred, [agent_states[-1][0], agent_states[-1][1], \
                np.arctan2(agent_states[-1][1] - agent_states[-2][1], agent_states[-1][0] - agent_states[-2][0])])
            ax.plot(x_pred, y_pred, color=base_color, alpha=0.6, marker='x')

        

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