'''
Code which implements a prioritized planner for MAPF of 5 robots. 
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import heapq
import time

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
from rps.utilities.controllers import create_si_position_controller

# # =========================================================
# # SIMULATION PARAMETERS
# # =========================================================
N                   = 10
safety_radius       = 0.15
projection_distance = 0.03
waypoint_tolerance  = 0.035
grid_cell_size      = safety_radius + 2 * projection_distance + 0.03

# # =========================================================
# # ROBOTARIUM INITIALIZATION
# # =========================================================
# r = robotarium.Robotarium(number_of_robots=N, show_figure=False)

# # =========================================================
# # GOAL INITIALIZATION
# # =========================================================
# xybound = np.array([-1.0, 1.0, -0.8, 0.8])
# p_theta = 2 * np.pi * (np.arange(1, 2*N, 2) / (2*N))
# p_circ  = np.vstack([
#     np.hstack([xybound[1] * np.cos(p_theta), xybound[1] * np.cos(p_theta + np.pi)]),
#     np.hstack([xybound[3] * np.sin(p_theta), xybound[3] * np.sin(p_theta + np.pi)]),
# ])

# x_goal = p_circ[:, :N]
# flag   = 0   # 0 = first half of ellipse, 1 = second half
# count  = 0   # number of completed swaps
# print(p_circ[0, :N] , len(p_circ[0, :N]))


# # =========================================================
# # CONTROLLER AND SAFETY SETUP
# # =========================================================
# position_int = create_si_position_controller(
#     x_velocity_gain=2.0, y_velocity_gain=2.0, velocity_magnitude_limit=0.15)

# si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
#     projection_distance=projection_distance)

# uni_barrier_certificate = create_uni_barrier_certificate(
#     barrier_gain=500, safety_radius=safety_radius,
#     projection_distance=projection_distance)


# # Get initial poses before plotting so markers start in the right place
# poses = r.get_poses()

# xy = poses[:2, :] 

# print(xy.shape)


def discretize_environment(bounds, cell_size, safety_margin=0.0):
    
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4,):
        raise ValueError("bounds must be [x_min, x_max, y_min, y_max]")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if safety_margin < 0:
        raise ValueError("safety_margin cannot be negative")

    robot_radius = robotarium.ARobotarium.ROBOT_DIAMETER / 2
    x_min, x_max, y_min, y_max = bounds
    x_min += robot_radius + safety_margin
    x_max -= robot_radius + safety_margin
    y_min += robot_radius + safety_margin
    y_max -= robot_radius + safety_margin

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Safety margin leaves no usable arena")

    # Round up so the final column/row covers any leftover edge strip. The
    # last cells may be partial, but their centers remain inside the usable
    # arena and valid Robotarium start poses still map to a cell.
    n_cols = int(np.ceil((x_max - x_min) / cell_size))
    n_rows = int(np.ceil((y_max - y_min) / cell_size))
    if n_cols == 0 or n_rows == 0:
        raise ValueError("cell_size is too large for the usable arena")

    return {
        "rows": n_rows,
        "cols": n_cols,
        "cell_size": float(cell_size),
        "origin": np.array([x_min, y_min]),
        "blocked": np.zeros((n_rows, n_cols), dtype=bool),
    }


def plot_grid(grid, bounds):
    """Plot grid-cell boundaries over the Robotarium arena."""
    x_min, x_max, y_min, y_max = np.asarray(bounds, dtype=float)
    origin_x, origin_y = grid["origin"]
    cell_size = grid["cell_size"]
    n_rows, n_cols = grid["rows"], grid["cols"]
    extent = [origin_x, origin_x + n_cols * cell_size,
              origin_y, origin_y + n_rows * cell_size]

    fig, ax = plt.subplots()
    ax.imshow(grid["blocked"], origin="lower", extent=extent,
              cmap="Greys", vmin=0, vmax=1, alpha=0.35,
              interpolation="none", aspect="equal")
    ax.set_xticks(origin_x + np.arange(n_cols + 1) * cell_size, minor=True)
    ax.set_yticks(origin_y + np.arange(n_rows + 1) * cell_size, minor=True)
    ax.grid(which="minor", color="steelblue", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Robotarium grid ({cell_size:.2f} m cells)")
    return fig, ax


def world_to_grid(xy, grid):
    """Convert world-coordinate positions (2 x N) to (row, col) cells."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 1:
        if xy.shape != (2,):
            raise ValueError("a single position must have shape (2,)")
        xy = xy.reshape(2, 1)
    if xy.ndim != 2 or xy.shape[0] != 2:
        raise ValueError("positions must have shape (2, N)")

    origin_x, origin_y = grid["origin"]
    cols = np.floor((xy[0, :] - origin_x) / grid["cell_size"]).astype(int)
    rows = np.floor((xy[1, :] - origin_y) / grid["cell_size"]).astype(int)
    cells = np.column_stack((rows, cols))

    inside = ((rows >= 0) & (rows < grid["rows"]) &
              (cols >= 0) & (cols < grid["cols"]))
    if not np.all(inside):
        bad = np.flatnonzero(~inside).tolist()
        raise ValueError(f"positions {bad} fall outside the grid")
    if np.any(grid["blocked"][rows, cols]):
        bad = np.flatnonzero(grid["blocked"][rows, cols]).tolist()
        raise ValueError(f"positions {bad} fall in blocked cells")
    return cells


def choose_goal_cells(start_cells, grid, seed=0):
    """Choose distinct free goal cells, excluding every robot's start cell."""
    start_cells = np.asarray(start_cells, dtype=int)
    if start_cells.ndim != 2 or start_cells.shape[1] != 2:
        raise ValueError("start_cells must have shape (N, 2), as (row, col)")

    n_rows, n_cols = grid["rows"], grid["cols"]
    free_cells = np.argwhere(~grid["blocked"])
    start_set = {tuple(cell) for cell in start_cells}
    candidates = [tuple(cell) for cell in free_cells
                  if tuple(cell) not in start_set]
    if len(candidates) < len(start_cells):
        raise ValueError("not enough free non-start cells to assign unique goals")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=len(start_cells), replace=False)
    return np.asarray([candidates[i] for i in selected], dtype=int)


def astar_time_expanded(start, goal, grid, start_heading=0,
                        vertex_reservations=None, edge_reservations=None,
                        start_time=0, max_time=500, turn_steps=1,
                        move_steps=1):
    """Plan four-direction moves with explicit turn and travel durations.

    Headings are 0=east, 1=north, 2=west, 3=south. A move to any adjacent
    cell is allowed. If it requires a heading change, the robot spends
    quarter-turns in place before moving forward. `turn_steps` is the number
    of planner ticks for each 90-degree turn; `move_steps` is the travel time
    for one cell. The returned path contains (row, col, heading, time) at
    action completion points (and repeated cells during turns/waits).

    Reservations are (row, col, time) vertices and
    ((from_row, from_col), (to_row, to_col), arrival_time) directed edges.
    """
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    vertex_reservations = vertex_reservations or set()
    edge_reservations = edge_reservations or set()

    def in_bounds(cell):
        row, col = cell
        return 0 <= row < grid["rows"] and 0 <= col < grid["cols"]

    def is_free(cell):
        return in_bounds(cell) and not grid["blocked"][cell]

    if not is_free(start) or not is_free(goal):
        raise ValueError("start and goal must be in-bounds, unblocked cells")
    if start_heading not in (0, 1, 2, 3):
        raise ValueError("start_heading must be 0 (east), 1 (north), 2 (west), or 3 (south)")
    if not isinstance(turn_steps, int) or turn_steps < 1:
        raise ValueError("turn_steps must be a positive integer")
    if not isinstance(move_steps, int) or move_steps < 1:
        raise ValueError("move_steps must be a positive integer")

    # Manhattan distance times move duration is admissible; omitting turns
    # keeps this heuristic simple and guarantees it never overestimates.
    def heuristic(row, col):
        return (abs(row - goal[0]) + abs(col - goal[1])) * move_steps

    start_state = (start[0], start[1], int(start_heading), int(start_time))
    # Store (f = g + h, g, tie-breaker, state).
    frontier = [(heuristic(*start_state[:2]), 0, 0, start_state)]
    parent = {start_state: None}
    best_cost = {start_state: 0}
    tie_breaker = 0
    # Direction associated with each heading: east, north, west, south.
    directions = ((0, 1), (-1, 0), (0, -1), (1, 0))

    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best_cost.get(state):
            continue
        row, col, heading, time = state
        current = (row, col)

        # A robot stays at its goal after arrival, so that cell must remain
        # free of all already planned reservations through the horizon.
        goal_is_clear_to_horizon = not any(
            (goal[0], goal[1], future_time) in vertex_reservations
            for future_time in range(time + 1, start_time + max_time + 1)
        )
        if current == goal and goal_is_clear_to_horizon:
            path = []
            while state is not None:
                path.append(state)
                state = parent[state]
            return path[::-1]
        if time >= start_time + max_time:
            continue

        # Candidate translations in all four grid directions. Turning occurs
        # at the current cell and consumes time before the translation.
        for next_heading, (dr, dc) in enumerate(directions):
            quarter_turns = (next_heading - heading) % 4
            quarter_turns = min(quarter_turns, 4 - quarter_turns)
            turn_duration = quarter_turns * turn_steps
            next_time = time + turn_duration + move_steps
            nxt = (row + dr, col + dc)
            if next_time > start_time + max_time or not is_free(nxt):
                continue

            # The robot occupies its current cell throughout the turn, then
            # the destination during travel. Check every intermediate tick.
            if any((row, col, tick) in vertex_reservations
                   for tick in range(time + 1, time + turn_duration + 1)):
                continue
            # During travel the displayed robot remains in its source cell;
            # it occupies the destination at the scheduled arrival tick.
            if any((row, col, tick) in vertex_reservations
                   for tick in range(time + turn_duration + 1, next_time)):
                continue
            if (nxt[0], nxt[1], next_time) in vertex_reservations:
                continue
            if any((nxt, current, tick) in edge_reservations
                   for tick in range(time + turn_duration + 1, next_time + 1)):
                continue

            next_state = (nxt[0], nxt[1], next_heading, next_time)
            next_cost = cost + turn_duration + move_steps
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue
            best_cost[next_state] = next_cost
            parent[next_state] = state
            tie_breaker += 1
            priority = next_cost + heuristic(nxt[0], nxt[1])
            heapq.heappush(frontier, (priority, next_cost, tie_breaker, next_state))

        # Waiting is a one-tick action and keeps position and heading.
        next_time = time + 1
        if next_time <= start_time + max_time and (row, col, next_time) not in vertex_reservations:
            next_state = (row, col, heading, next_time)
            next_cost = cost + 1
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                parent[next_state] = state
                tie_breaker += 1
                priority = next_cost + heuristic(row, col)
                heapq.heappush(frontier, (priority, next_cost, tie_breaker, next_state))

    return None


def create_reservation_table():
    """Create empty vertex and directed-edge reservation sets."""
    return {"vertices": set(), "edges": set()}


def reserve_path(path, reservations, hold_goal_until=None):
    """Add a timed A* path to the reservation table.

    Each path state is (row, col, heading, time). Every cell occupied between
    consecutive states is reserved for each integer tick, including turns and
    waits. Translation edges are reserved at each tick spent traveling. If
    hold_goal_until is given, the final cell stays reserved through that time.
    """
    if not path:
        raise ValueError("cannot reserve an empty path")

    vertices = reservations["vertices"]
    edges = reservations["edges"]
    first_row, first_col, _, first_time = path[0]
    vertices.add((first_row, first_col, first_time))

    for previous, current in zip(path, path[1:]):
        prev_row, prev_col, _, prev_time = previous
        row, col, _, time = current
        prev_cell = (prev_row, prev_col)
        cell = (row, col)

        if cell == prev_cell:
            # Turn or wait: robot remains in its cell for the interval.
            for tick in range(prev_time + 1, time + 1):
                vertices.add((row, col, tick))
        else:
            # Match the discrete animation model: the robot stays in its
            # source cell until arrival, then occupies the destination.
            for tick in range(prev_time + 1, time):
                vertices.add((prev_row, prev_col, tick))
                edges.add((prev_cell, cell, tick))
            vertices.add((row, col, time))
            edges.add((prev_cell, cell, time))

    final_row, final_col, _, final_time = path[-1]
    if hold_goal_until is not None:
        for tick in range(final_time + 1, int(hold_goal_until) + 1):
            vertices.add((final_row, final_col, tick))


def plan_prioritized(start_cells, goal_cells, start_headings, grid, priorities,
                     max_time=500, turn_steps=1, move_steps=1):
    """Plan robots by priority, reserving each successful path for the next."""
    starts = np.asarray(start_cells, dtype=int)
    goals = np.asarray(goal_cells, dtype=int)
    headings = np.asarray(start_headings, dtype=int)
    if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("start_cells and goal_cells must both have shape (N, 2)")
    if headings.shape != (len(starts),):
        raise ValueError("start_headings must have one heading per robot")
    if sorted(priorities) != list(range(len(starts))):
        raise ValueError("priorities must be a permutation of robot indices")

    reservations = create_reservation_table()
    paths = [None] * len(starts)
    for robot_id in priorities:
        path = astar_time_expanded(
            starts[robot_id], goals[robot_id], grid,
            start_heading=int(headings[robot_id]),
            vertex_reservations=reservations["vertices"],
            edge_reservations=reservations["edges"],
            max_time=max_time, turn_steps=turn_steps,
            move_steps=move_steps,
        )
        if path is None:
            return None, reservations, robot_id
        paths[robot_id] = path
        reserve_path(path, reservations, hold_goal_until=max_time)
    return paths, reservations, None


def animate_paths(paths, grid, bounds, interval=0.2):
    """Animate timed grid paths with matplotlib only (no Robotarium needed)."""
    if not paths:
        raise ValueError("paths must contain at least one robot path")
    max_time = max(path[-1][3] for path in paths)
    colors = plt.get_cmap("tab10")
    origin_x, origin_y = grid["origin"]
    cell_size = grid["cell_size"]

    def cell_center(cell):
        row, col = cell
        return (origin_x + (col + 0.5) * cell_size,
                origin_y + (row + 0.5) * cell_size)

    def cell_at_time(path, time_step):
        # Show the discrete MAPF occupancy: turns/waits stay at the previous
        # cell; translation changes cells only at its scheduled arrival tick.
        for previous, current in zip(path, path[1:]):
            r0, c0, _, t0 = previous
            r1, c1, _, t1 = current
            if time_step <= t1:
                cell = (r0, c0) if time_step < t1 else (r1, c1)
                return np.asarray(cell_center(cell))
        return np.asarray(cell_center(path[-1][:2]))

    plt.ion()
    fig, ax = plot_grid(grid, bounds)
    starts_xy = np.array([cell_center(path[0][:2]) for path in paths])
    goals_xy = np.array([cell_center(path[-1][:2]) for path in paths])
    ax.scatter(starts_xy[:, 0], starts_xy[:, 1], marker="o", facecolors="none",
               edgecolors=[colors(i % 10) for i in range(len(paths))],
               label="starts")
    ax.scatter(goals_xy[:, 0], goals_xy[:, 1], marker="x",
               color=[colors(i % 10) for i in range(len(paths))], label="goals")
    robots = ax.scatter(starts_xy[:, 0], starts_xy[:, 1], s=90,
                        c=[colors(i % 10) for i in range(len(paths))],
                        edgecolors="black", zorder=5)
    ax.legend(loc="upper right")
    plt.show(block=False)

    for time_step in range(max_time + 1):
        positions = np.array([cell_at_time(path, time_step) for path in paths])
        occupied = [tuple(path_cell) for path_cell in
                    (state_at_time(path, time_step) for path in paths)]
        if len(set(occupied)) != len(occupied):
            raise RuntimeError(f"Planned paths collide at time step {time_step}: {occupied}")
        robots.set_offsets(positions)
        ax.set_title(f"Prioritized MAPF paths — time step {time_step}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(interval)

    plt.ioff()
    plt.show()
    return fig, ax


def state_at_time(path, time_step):
    """Return the occupied grid cell for a path at an integer time step."""
    for previous, current in zip(path, path[1:]):
        r0, c0, _, _ = previous
        r1, c1, _, t1 = current
        if time_step < t1:
            return (r0, c0)
        if time_step == t1:
            return (r1, c1)
    return tuple(path[-1][:2])


def cell_to_world(cell, grid):
    """Return the world-coordinate center of a (row, col) grid cell."""
    row, col = cell
    return np.array([
        grid["origin"][0] + (col + 0.5) * grid["cell_size"],
        grid["origin"][1] + (row + 0.5) * grid["cell_size"],
    ])


def run_robotarium(r, paths, grid, start_poses, max_iterations=30000):
    """Execute the planned cell routes with Robotarium's controllers.

    The A* reservation schedule supplies routes and ordering, while the
    controller follows each route based on actual pose feedback. The
    unicycle barrier certificate provides reactive collision protection if
    robots arrive later than their discrete plan predicts.
    """
    n_robots = len(paths)
    ax = r._axes_handle
    fig = r._fig
    colors = plt.get_cmap("tab10")

    # Draw the MAPF grid and planned paths on the Robotarium's own axes.
    origin_x, origin_y = grid["origin"]
    cell_size = grid["cell_size"]
    for col in range(grid["cols"] + 1):
        x = origin_x + col * cell_size
        ax.axvline(x, color="steelblue", linewidth=0.45, alpha=0.3, zorder=0)
    for row in range(grid["rows"] + 1):
        y = origin_y + row * cell_size
        ax.axhline(y, color="steelblue", linewidth=0.45, alpha=0.3, zorder=0)

    planned_lines = []
    actual_lines = []
    for robot_id, path in enumerate(paths):
        color = colors(robot_id % 10)
        path_cells = []
        for row, col, _, _ in path:
            cell = (row, col)
            if not path_cells or cell != path_cells[-1]:
                path_cells.append(cell)
        path_xy = np.array([cell_to_world(cell, grid) for cell in path_cells])
        planned_line, = ax.plot(path_xy[:, 0], path_xy[:, 1], "--",
                                color=color, linewidth=1.6, alpha=0.75,
                                zorder=1)
        actual_line, = ax.plot([], [], "-", color=color, linewidth=2.2,
                               zorder=3)
        planned_lines.append(planned_line)
        actual_lines.append(actual_line)

        # Start marker uses the actual Robotarium pose; goal is the final cell
        # center. Robot patches remain visible above these overlays.
        ax.scatter(start_poses[0, robot_id], start_poses[1, robot_id],
                   marker="o", s=70, facecolors="none", edgecolors=[color],
                   linewidths=1.8, zorder=4)
        goal_xy = cell_to_world(path_cells[-1], grid)
        ax.scatter(goal_xy[0], goal_xy[1], marker="x", s=70, color=[color],
                   linewidths=2.0, zorder=4)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6,
               label="planned MAPF path"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.2,
               label="controller trail"),
        Line2D([0], [0], color="black", marker="o", linestyle="None",
               markerfacecolor="none", label="start"),
        Line2D([0], [0], color="black", marker="x", linestyle="None",
               label="goal"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title("MAPF planned routes and controller trajectories")
    fig.canvas.draw_idle()

    position_controller = create_si_position_controller(
        x_velocity_gain=2.0, y_velocity_gain=2.0,
        velocity_magnitude_limit=0.15,
    )
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
        projection_distance=projection_distance,
    )
    barrier_certificate = create_uni_barrier_certificate(
        barrier_gain=100,
        safety_radius=safety_radius,
        projection_distance=projection_distance,
    )

    # Collapse turn/wait states into the sequence of distinct cell targets.
    waypoint_cells = []
    for path in paths:
        cells = []
        for row, col, _, _ in path:
            cell = (row, col)
            if not cells or cell != cells[-1]:
                cells.append(cell)
        waypoint_cells.append(cells)
    waypoint_indices = np.zeros(n_robots, dtype=int)
    finished = np.zeros(n_robots, dtype=bool)
    robot_ids = np.arange(n_robots)

    for iteration in range(max_iterations):
        poses = r.get_poses()
        xi = uni_to_si_states(poses)
        for robot_id, line in enumerate(actual_lines):
            line_x = np.append(line.get_xdata(), poses[0, robot_id])
            line_y = np.append(line.get_ydata(), poses[1, robot_id])
            line.set_data(line_x, line_y)
        targets = np.zeros((2, n_robots))

        for robot_id in range(n_robots):
            cells = waypoint_cells[robot_id]
            while not finished[robot_id]:
                target = cell_to_world(cells[waypoint_indices[robot_id]], grid)
                if np.linalg.norm(xi[:, robot_id] - target) > waypoint_tolerance:
                    targets[:, robot_id] = target
                    break
                waypoint_indices[robot_id] += 1
                if waypoint_indices[robot_id] >= len(cells):
                    finished[robot_id] = True
            if finished[robot_id]:
                targets[:, robot_id] = xi[:, robot_id]

        if np.all(finished):
            print(f"All {n_robots} robots reached their goals in {iteration} steps.")
            break

        dxi = position_controller(xi, targets)
        for robot_id in np.flatnonzero(finished):
            dxi[:, robot_id] = 0.0
        dxu = si_to_uni_dyn(dxi, poses)
        dxu = barrier_certificate(dxu, poses)
        r.set_velocities(robot_ids, dxu)
        r.step()
        ax.set_title(
            f"MAPF planned routes and controller trajectories — step {iteration}"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    else:
        raise RuntimeError(
            f"Execution exceeded {max_iterations} simulator steps; "
            f"finished {int(np.sum(finished))}/{n_robots} robots."
        )

    r.debug()
    return r


grid = discretize_environment(
    robotarium.ARobotarium.BOUNDARIES,
    # Neighboring cell centers should be farther apart than the barrier's
    # effective clearance (safety_radius + 2 * projection_distance).
    cell_size=grid_cell_size,
    safety_margin=0.0,
)

priorities = list(range(N)) 

# Initialize Robotarium, convert its actual starting poses to grid cells, and
# choose distinct goal cells that aren't occupied by any robot's start.
r = robotarium.Robotarium(number_of_robots=N, show_figure=True)
initial_poses = r.get_poses()
start_cells = world_to_grid(initial_poses[:2, :], grid)
start_headings = np.mod(
    np.rint(initial_poses[2, :] / (np.pi / 2)).astype(int), 4
)
goal_cells = choose_goal_cells(start_cells, grid, seed=0)

paths, reservations, failed_robot = plan_prioritized(
    start_cells, goal_cells, start_headings, grid, priorities,
    max_time=500, turn_steps=1, move_steps=1,
)
if paths is None:
    raise RuntimeError(f"Prioritized planning failed for robot {failed_robot}")


print(f"Grid: {grid['cols']} columns x {grid['rows']} rows; origin={grid['origin']}")
print("Start cells (row, col):\n", start_cells)
print("Goal cells (row, col):\n", goal_cells)
print("Start headings (0=east, 1=north, 2=west, 3=south):\n", start_headings)
print("Planned arrival times (planner ticks):", [path[-1][3] for path in paths])

# Complete the pose read used for planning before entering the control loop.
r.step()
run_robotarium(r, paths, grid, initial_poses)







