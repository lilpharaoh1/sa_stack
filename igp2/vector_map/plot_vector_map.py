import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrow
import numpy as np

from igp2.opendrive import Map
from igp2.opendrive.elements.road_lanes import LaneTypes

try:
    from .dataset import Dataset
except:
    from dataset import Dataset

def world_to_ego_batch(xs, ys, agent_pose):
    cx, cy, heading = agent_pose
    dx = np.array(xs) - cx
    dy = np.array(ys) - cy
    theta = -(heading - np.pi/2)
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    x_ego = cos_h * dx - sin_h * dy
    y_ego = sin_h * dx + cos_h * dy
    return x_ego, y_ego

def plot_vector_map(dataset, odr_map, ax: plt.Axes = None, scenario_config=None, **kwargs) -> plt.Axes:
    """Plot OpenDRIVE map, optionally transformed to the ego frame."""
    colors = plt.get_cmap("tab10").colors
    transform = dataset.agent is not None  # Whether to transform to ego frame

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(odr_map.name)

    if odr_map is None:
        return ax

    ax.set_facecolor("grey")
    ax.set_aspect("equal")

    if kwargs.get("plot_background", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw background")
        background_path = f"{scenario_config.data_root}/{scenario_config.background_image}"
        background = imageio.imread(background_path)
        rescale = scenario_config.background_px_to_meter
        extent = (0, background.shape[1] * rescale, -background.shape[0] * rescale, 0)
        ax.imshow(background, extent=extent)

    if kwargs.get("plot_buildings", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw buildings")
        for building in scenario_config.buildings:
            building.append(building[0])
            x, y = zip(*building)
            if transform:
                x, y = world_to_ego_batch(x, y, dataset.agent)
            ax.plot(x, y, color="black")

    if kwargs.get("plot_goals", False):
        if scenario_config is None:
            raise ValueError("scenario_config must be provided to draw goals")
        for goal in scenario_config.goals:
            x, y = goal
            if transform:
                x, y = world_to_ego_batch([x], [y], dataset.agent)
            ax.plot(x, y, "ro", ms=10)

    if kwargs.get("ignore_roads", False):
        return ax

    # --- Plot Roads ---
    for road_id, road in odr_map.roads.items():
        boundary = road.boundary.boundary
        color = kwargs.get("road_color", "k")

        # Plot road boundaries
        if boundary.geom_type == "LineString":
            x, y = boundary.xy
            if transform:
                x, y = world_to_ego_batch(x, y, dataset.agent)
            ax.plot(x, y, color=color)
        elif boundary.geom_type == "MultiLineString":
            for b in boundary.geoms:
                x, y = b.xy
                if transform:
                    x, y = world_to_ego_batch(x, y, dataset.agent)
                ax.plot(x, y, color="orange")

        # Plot midlines
        if kwargs.get("midline", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0:
                        continue
                    if kwargs.get("drivable", False) and lane.type != "driving":
                        continue

                    x, y = lane.midline.xy
                    if transform:
                        x, y = world_to_ego_batch(x, y, dataset.agent)

                    if kwargs.get("midline_direction", False):
                        ax.quiver(
                            x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                            width=0.0025, headwidth=2,
                            scale_units='xy', angles='xy', scale=1, color="red"
                        )
                    else:
                        ax.plot(x, y, color=kwargs.get("midline_color", "r"))

        # Road IDs
        if kwargs.get("road_ids", False):
            midx = np.mean(road.midline.xy[0])
            midy = np.mean(road.midline.xy[1])
            if transform:
                midx, midy = world_to_ego_batch([midx], [midy], dataset.agent)
            ax.text(midx, midy, str(road.id),
                    color=colors[road_id % len(colors)], fontsize=10)

        # Lane markings
        if kwargs.get("markings", False):
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    for marker in lane.markers:
                        for i, style in enumerate(marker.type_to_linestyle):
                            if style is None:
                                continue
                            df = 0.13
                            side = "left" if lane.id <= 0 else "right"
                            line = lane.reference_line.parallel_offset(i * df, side=side)
                            x, y = line.xy
                            if transform:
                                x, y = world_to_ego_batch(x, y, dataset.agent)
                            ax.plot(x, y,
                                    color=marker.color_to_rgb,
                                    linestyle=style,
                                    linewidth=marker.plot_width)

    # --- Plot Junctions ---
    for junction_id, junction in odr_map.junctions.items():
        if hasattr(junction.boundary, "geoms"):
            polys = junction.boundary.geoms
        else:
            polys = [junction.boundary]

        for poly in polys:
            x, y = poly.boundary.xy
            if transform:
                x, y = world_to_ego_batch(x, y, dataset.agent)
            ax.fill(
                x, y,
                color=kwargs.get("junction_color", (0.941, 1.0, 0.420, 0.5))
            )

    # --- Plot Lane Graph (already ego-frame) ---
    for node_id, node in dataset.nodes.items():
        x, y = node["pose"]
        ax.plot(x, y, "bo")
        ax.text(x, y, node_id, fontsize=20)

    for start_id, end_id in dataset.edges:
        s = dataset.nodes[start_id]["pose"]
        e = dataset.nodes[end_id]["pose"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        ax.arrow(s[0], s[1], dx, dy, head_width=1, width=0.5, length_includes_head=True)

    # --- Draw ego box ---
    if dataset.agent and dataset.bounds and kwargs.get("agent", False):
        cx, cy, heading = dataset.agent
        left, right, back, front = dataset.bounds
        corners_local = np.array([[front, left], [front, right], [back, right], [back, left]])
        # Plot as polygon
        poly = Polygon(corners_local, closed=True, alpha=0.2)
        ax.add_patch(poly)

        # Vehicle dimensions (in meters)
        width = 2.0   # side-to-side
        length = 4.5  # front-to-back

        # Draw rectangle centered at origin
        rect = Rectangle((-length/2, -width/2), length, width,
                        edgecolor='red', facecolor='none', linestyle='--', linewidth=2)
        ax.add_patch(rect)

        # Draw arrow pointing forward (right, in ego frame)
        arrow = FancyArrow(0, 0, length/2, 0,
                        width=0.5, head_width=1.0, head_length=1.0,
                        color='RED')
        ax.add_patch(arrow)

    return ax


if __name__ == '__main__':
    xodr_file = f"scenarios/maps/scenario1.xodr"
    odr_map = Map.parse_from_opendrive(xodr_file)
    dataset = Dataset.parse_from_opendrive(xodr_file)
    # dataset.generate_graph(agent_pose=[90, -75, np.pi / 2])
    # dataset.generate_graph(agent_pose=[35.0, -1.8, np.pi / 2])
    dataset.generate_graph(agent_pose=[52.0, -42.5, np.pi / 2])
    # plot_map(odr_map, markings=True, midline=True)
    plot_vector_map(dataset, odr_map, markings=True, agent=True)
    
    plt.show()
