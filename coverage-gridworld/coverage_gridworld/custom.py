import numpy as np
import gymnasium as gym
from collections import deque

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

#Choosing Observation Mode
# 1 = full grid
# 2 = compact (3 features)
# 3 = local patch + global metrics
# 4 = frontier/risk engineered features
# 5 = local patch + explicit enemy FOV prediction (next / +2 rotations) + per-action lookahead
OBS_MODE = 5

# Choosing reward mode (used by reward(); same idea as OBS_MODE for observation_space / observation).
# 1 = balanced, 2 = safety-prioritized, 3 = greedy, 4 = enhanced (BFS, FOV lookahead, anti-loop)
REWARD_MODE = 4

# Color constants 
_BLACK      = (0,   0,   0)    # unexplored cell
_WHITE      = (255, 255, 255)  # explored cell
_BROWN      = (101, 67,  33)   # wall
_GREY       = (160, 161, 161)  # agent
_GREEN      = (31,  198, 0)    # enemy
_RED        = (255, 0,   0)    # unexplored cell under enemy FOV
_LIGHT_RED  = (255, 127, 127)  # explored cell under enemy FOV
 
# Anti-loop trackers for reward shaping (module-level across steps).
_LAST_AGENT_POS = None
_PREV_AGENT_POS = None
_STILL_COUNT = 0
_TWO_STEP_LOOP_COUNT = 0
_LAST_STEPS_REMAINING = None
_PREV_SAFE_FRONTIER_DIST = None
_CURR_NEAREST_UNEXPLORED_DIST = 20.0
_VISITED_POSITIONS = set()
_NO_PROGRESS_STREAK = 0
_WALL_CELLS = set()
_RECENT_POSITIONS = []
_RECENT_WINDOW = 14
_STEPS_SINCE_DISCOVERY = 0  # steps elapsed since the last new cell was covered
_NEXT_FOV_CELLS: set = set()  # enemy FOV 2 rotations ahead (= 1 step ahead by the time obs reads it)
# Post-step grid copy for reward_func 4 BFS shaping (set in observation() each step).
_GRID_SNAPSHOT_FOR_REWARD: np.ndarray | None = None
# Previous BFS shortest-path distance to nearest unexplored (reward_func 4); reset each episode.
_PREV_BFS_DIST: float | None = None

# reward_func == 4: potential-based-style shaping toward walkable frontier (clip per step).
REWARD4_BFS_ALPHA = 0.35
REWARD4_BFS_SHAPING_CAP = 1.5
# Extra exploration when the agent is stuck at partial coverage (e.g. ~0.7 with no full clear).
_REWARD4_EPISODE_VISITED: set[int] = set()  # distinct grid positions visited this episode
REWARD4_FIRST_VISIT_BONUS = 0.18  # once per agent_pos per episode → spread paths, break loops
REWARD4_FRONTIER_ADJ_BONUS = 0.07  # per step while orthogonally adjacent to BLACK/RED (unexplored)
# Below this coverage ratio, new cells get +REWARD4_LOW_COVERAGE_NEWCELL_EXTRA.  Using ~0.72
# matched an early design point but matched a policy plateau (~0.72 on chokepoint): the bonus
# vanished right when the last corridor cells are hardest.  Keep high so endgame still pays.
REWARD4_LOW_COVERAGE_THRESHOLD = 0.90
REWARD4_LOW_COVERAGE_NEWCELL_EXTRA = 2.5  # on top of +5 when coverage below threshold
REWARD4_EXPLORE_BOOST_COVER_MAX = 0.88  # scale BFS shaping while coverage is below this
REWARD4_EXPLORE_BOOST_BFS = 1.25  # multiplicative boost to BFS alpha when coverage is low (up to 1+this)
# Danger: stronger signal so exploration shaping does not drown "do not enter the cone"
REWARD4_IMMEDIATE_DANGER_PENALTY = 36.0
REWARD4_CURRENTLY_SEEN_PENALTY = 14.0
REWARD4_GAME_OVER_PENALTY = -88.0
# Terminal coverage bonus: added at episode end (death or timeout) so that
# dying at 75% coverage is better than dying at 55%.  Without this the agent
# converges to a safe-but-limited corridor and never attempts harder sections.
REWARD4_TERMINAL_COVERAGE_BONUS = 120.0  # multiplied by coverage_ratio at termination
# When next enemy rotation would sweep the agent's tile, scale down the big new-cell
# bonuses (+5 / low-cov / unseen) so a reckless new tile cannot outweigh the hazard signal.
REWARD4_NEWCELL_DANGER_SCALE = 0.32
# While in enemy FOV or about to enter next-step FOV, scale down auxiliary explore bonuses
REWARD4_UNSAFE_EXPLORE_SCALE = 0.12
# Many-enemy maps (e.g. sneaky_enemies): next-step FOV is often true; use milder suppression
REWARD4_MANY_ENEMIES_THRESHOLD = 5
REWARD4_UNSAFE_EXPLORE_SCALE_DENSE = 0.38
# Clip dense per-step return (before +200 victory) so critic scale stays stable
REWARD4_STEP_CLIP_MIN = -62.0
REWARD4_STEP_CLIP_MAX = 24.0



#Helper function to get the type of the cell
def cell_type(rgb: np.ndarray) -> int:
    """Map an RGB triple to a compact integer cell-type label (0-6)."""
    rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    mapping = {
        _BLACK:     0,
        _WHITE:     1,
        _BROWN:     2,
        _GREY:      3,
        _GREEN:     4,
        _RED:       5,
        _LIGHT_RED: 6,
    }
    return mapping.get(rgb, 0)



# OBSERVATION SPACE 1 — Full flattened grid 

def obs_space_full(env: gym.Env) -> gym.spaces.Space:
    """
    Each of the 10×10 = 100 cells is encoded as a single integer in [0, 6]
    representing the cell type (unexplored, explored, wall, agent, enemy,
    danger-unexplored, danger-explored).  Total: 100-dim MultiDiscrete.
    """
    return gym.spaces.MultiDiscrete([7] * (env.grid_size * env.grid_size))
 
 
def obs_full(grid: np.ndarray) -> np.ndarray:
    """Return a flat integer array of shape (100,), one label per cell."""
    rows, cols = grid.shape[:2]
    out = np.zeros(rows * cols, dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            out[r * cols + c] = cell_type(grid[r, c])
    return out


# OBSERVATION SPACE 2 — Compact feature vector (Box, 3 dims)
def obs_space_compact(env: gym.Env) -> gym.spaces.Space:
    """
    A lightweight 3-dimensional Box observation, all values normalised to [0, 1]:
      [0] agent_position  — flat grid index normalised by total cells (0 to 1)
      [1] coverage_ratio  — how much of the map has been explored (0 to 1)
      [2] danger_ratio    — fraction of cells currently under enemy FOV (0 to 1)
    """
    return gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
 
 
def obs_compact(grid: np.ndarray) -> np.ndarray:
    """Return the 3-dim feature vector from the raw grid."""
    rows, cols = grid.shape[:2]
    n_cells = rows * cols
 
    agent_pos_flat = 0
    coverable      = 0
    covered        = 0
    danger_cells   = 0
 
    for r in range(rows):
        for c in range(cols):
            ct = cell_type(grid[r, c])
            if ct == 3:              # GREY = agent
                agent_pos_flat = r * cols + c
            if ct != 2:              # not a wall → coverable
                coverable += 1
            if ct in (1, 3, 6):     # WHITE, GREY, LIGHT_RED = already explored
                covered += 1
            if ct in (5, 6):         # RED, LIGHT_RED = under enemy FOV
                danger_cells += 1
 
    return np.array([
        agent_pos_flat / n_cells,           # [0] normalised position
        covered / max(coverable, 1),        # [1] coverage ratio
        danger_cells / n_cells,             # [2] danger ratio
    ], dtype=np.float32)


def _bfs_nearest_unexplored(grid: np.ndarray, agent_r: int, agent_c: int):
    """
    BFS from the agent's cell to find the nearest unexplored cell, respecting
    walls. Returns (path_length, first_step_dr, first_step_dc).

    The first-step direction tells the agent exactly which move to make next
    on the shortest wall-aware path — unlike Manhattan direction, which points
    straight through walls and misleads the policy in walled environments.

    Returns (0, 0, 0) when no unexplored cell is reachable (map fully covered).
    """
    rows, cols = grid.shape[:2]
    visited = {(agent_r, agent_c)}
    # Seed the queue with the four immediate neighbours, tagged with the
    # first-step direction used to reach them.
    queue = deque()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_r + dr, agent_c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            ct = cell_type(grid[nr, nc])
            if ct == 2:          # wall — skip
                continue
            visited.add((nr, nc))
            if ct in (0, 5):     # unexplored cell found 1 step away
                return 1, dr, dc
            queue.append((nr, nc, dr, dc, 1))

    while queue:
        r, c, fdr, fdc, dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                ct = cell_type(grid[nr, nc])
                if ct == 2:
                    continue
                visited.add((nr, nc))
                if ct in (0, 5):
                    return dist + 1, fdr, fdc
                queue.append((nr, nc, fdr, fdc, dist + 1))

    return 0, 0, 0   # fully explored or no reachable unexplored cell


def _agent_adjacent_unexplored(grid: np.ndarray, ar: int, ac: int) -> bool:
    """True if any 4-neighbor of the agent is still unexplored (BLACK or RED)."""
    rows, cols = grid.shape[:2]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = ar + dr, ac + dc
        if 0 <= r < rows and 0 <= c < cols and cell_type(grid[r, c]) in (0, 5):
            return True
    return False


def _grid_metrics(grid: np.ndarray):
    """Collect reusable grid statistics for engineered observations."""
    rows, cols = grid.shape[:2]
    n_cells = rows * cols
    max_manhattan = max(rows + cols - 2, 1)

    agent_r, agent_c = 0, 0
    coverable = 0
    covered = 0
    danger_cells = 0
    unexplored = []
    danger_positions = []

    for r in range(rows):
        for c in range(cols):
            ct = cell_type(grid[r, c])
            if ct == 3:
                agent_r, agent_c = r, c
            if ct != 2:
                coverable += 1
            if ct in (1, 3, 6):
                covered += 1
            if ct in (5, 6):
                danger_cells += 1
                danger_positions.append((r, c))
            if ct in (0, 5):
                unexplored.append((r, c))

    return {
        "rows": rows,
        "cols": cols,
        "n_cells": n_cells,
        "max_manhattan": max_manhattan,
        "agent_r": agent_r,
        "agent_c": agent_c,
        "coverable": coverable,
        "covered": covered,
        "danger_cells": danger_cells,
        "unexplored": unexplored,
        "danger_positions": danger_positions,
    }


# OBSERVATION SPACE 3 — Local patch + global metrics
PATCH_RADIUS = 3  # 7x7 patch around the agent
LOCAL_PATCH_NUM_CELLS = (2 * PATCH_RADIUS + 1) ** 2
# patch (49) + 5 globals + 5 action-safety flags (grid-inferred, no phase delay)
OBS3_DIM = LOCAL_PATCH_NUM_CELLS + 10


def obs_space_local_patch(env: gym.Env) -> gym.spaces.Space:
    return gym.spaces.Box(low=0.0, high=1.0, shape=(OBS3_DIM,), dtype=np.float32)


def obs_local_patch(grid: np.ndarray) -> np.ndarray:
    global _CURR_NEAREST_UNEXPLORED_DIST, _NEXT_FOV_CELLS, _WALL_CELLS
    metrics = _grid_metrics(grid)
    rows = metrics["rows"]
    cols = metrics["cols"]
    agent_r = metrics["agent_r"]
    agent_c = metrics["agent_c"]
    max_manhattan = metrics["max_manhattan"]

    patch_vals = []
    for dr in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
        for dc in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            rr = agent_r + dr
            cc = agent_c + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                # out-of-bounds treated as wall for local navigation
                patch_vals.append(2.0 / 6.0)
            else:
                patch_vals.append(cell_type(grid[rr, cc]) / 6.0)
    assert len(patch_vals) == LOCAL_PATCH_NUM_CELLS

    coverage_ratio = metrics["covered"] / max(metrics["coverable"], 1)
    danger_ratio = metrics["danger_cells"] / metrics["n_cells"]

    # BFS to nearest unexplored — path-aware, respects walls.
    # Encodes the FIRST STEP of the shortest walkable path, not the Manhattan
    # direction, so the hint is never misleading in walled environments.
    #   dir_row: 0.0 = step up, 0.5 = no vertical component, 1.0 = step down
    #   dir_col: 0.0 = step left, 0.5 = no horizontal component, 1.0 = step right
    bfs_dist, bfs_dr, bfs_dc = _bfs_nearest_unexplored(grid, agent_r, agent_c)
    _CURR_NEAREST_UNEXPLORED_DIST = float(bfs_dist)
    if bfs_dist > 0:
        dir_row = float(bfs_dr + 1) / 2.0   # -1→0.0, 0→0.5, +1→1.0
        dir_col = float(bfs_dc + 1) / 2.0
    else:
        dir_row = 0.5   # fully explored
        dir_col = 0.5

    if metrics["danger_positions"]:
        nearest_danger = min(
            abs(agent_r - r) + abs(agent_c - c) for (r, c) in metrics["danger_positions"]
        )
        nearest_danger /= max_manhattan
    else:
        nearest_danger = 1.0

    # 5 action-safety features [LEFT, DOWN, RIGHT, UP, STAY]:
    # 1.0 = destination will NOT be in enemy FOV after the next rotation.
    # 0.0 = destination will be in FOV (or blocked → forced stay is unsafe).
    # Computed fresh from the grid each step (infer orientations, predict +1 rotation)
    # so the signal is never stale — the old _NEXT_FOV_CELLS cache had a phase delay
    # that made step 0 blind and every step see slightly wrong timing.
    enemy_cells_obs = sorted(
        (r, c) for r in range(rows) for c in range(cols) if cell_type(grid[r, c]) == 4
    )
    walls_obs = set(_WALL_CELLS)
    enemy_bodies_obs = set(enemy_cells_obs)
    if enemy_cells_obs:
        orientations_obs = _infer_enemy_orientations(
            grid, enemy_cells_obs, walls_obs, enemy_bodies_obs, rows, cols
        )
        fov_next_obs = _union_predicted_fov(
            enemy_cells_obs, orientations_obs, 1, walls_obs, enemy_bodies_obs, rows, cols
        )
    else:
        fov_next_obs = set()

    action_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]  # L,D,R,U,Stay
    action_safe = []
    for dr, dc in action_deltas:
        nr, nc = agent_r + dr, agent_c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls_obs:
            dest = (agent_r, agent_c)  # blocked → agent stays
        else:
            dest = (nr, nc)
        action_safe.append(0.0 if dest in fov_next_obs else 1.0)

    out = np.array(
        patch_vals + [coverage_ratio, danger_ratio, dir_row, dir_col, nearest_danger] + action_safe,
        dtype=np.float32,
    )
    assert out.shape[0] == OBS3_DIM
    return out


# OBSERVATION SPACE 4 — Frontier/risk engineered features (Box, 12 dims)
def obs_space_frontier_risk(_: gym.Env) -> gym.spaces.Space:
    return gym.spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)


def obs_frontier_risk(grid: np.ndarray) -> np.ndarray:
    metrics = _grid_metrics(grid)
    rows = metrics["rows"]
    cols = metrics["cols"]
    agent_r = metrics["agent_r"]
    agent_c = metrics["agent_c"]
    coverable = max(metrics["coverable"], 1)
    max_manhattan = metrics["max_manhattan"]
    danger_set = set(metrics["danger_positions"])

    explored_types = (1, 3, 6)
    unexplored_types = (0, 5)
    frontier_cells = []
    safe_frontier_cells = []

    for r in range(rows):
        for c in range(cols):
            ct = cell_type(grid[r, c])
            if ct not in unexplored_types:
                continue
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            has_explored_neighbor = False
            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:
                    if cell_type(grid[nr, nc]) in explored_types:
                        has_explored_neighbor = True
                        break
            if has_explored_neighbor:
                frontier_cells.append((r, c))
                if (r, c) not in danger_set:
                    safe_frontier_cells.append((r, c))

    if frontier_cells:
        nearest_frontier = min(abs(agent_r - r) + abs(agent_c - c) for (r, c) in frontier_cells)
        nearest_frontier /= max_manhattan
    else:
        nearest_frontier = 1.0

    if safe_frontier_cells:
        nearest_safe_frontier = min(
            abs(agent_r - r) + abs(agent_c - c) for (r, c) in safe_frontier_cells
        )
        nearest_safe_frontier /= max_manhattan
    else:
        nearest_safe_frontier = 1.0

    legal_moves = 0
    safe_moves = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = agent_r + dr
        cc = agent_c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            if cell_type(grid[rr, cc]) != 2:  # not wall
                legal_moves += 1
                if (rr, cc) not in danger_set:
                    safe_moves += 1

    # local 5x5 summaries around the agent
    local_total = 0
    local_walls = 0
    local_danger = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = agent_r + dr
            cc = agent_c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                local_total += 1
                ct = cell_type(grid[rr, cc])
                if ct == 2:
                    local_walls += 1
                if ct in (5, 6):
                    local_danger += 1

    feature_vec = np.array(
        [
            agent_r / max(rows - 1, 1),
            agent_c / max(cols - 1, 1),
            metrics["covered"] / coverable,
            metrics["danger_cells"] / metrics["n_cells"],
            len(metrics["unexplored"]) / coverable,
            len(frontier_cells) / coverable,
            len(safe_frontier_cells) / coverable,
            nearest_frontier,
            nearest_safe_frontier,
            legal_moves / 4.0,
            safe_moves / 4.0,
            (local_danger / max(local_total, 1)),
        ],
        dtype=np.float32,
    )

    return feature_vec


# --- OBSERVATION SPACE 5 — Fork of mode 3 + explicit enemy FOV prediction tail ---
# Prefix is exactly obs_local_patch (OBS3_DIM). Suffix adds inferred-orientation FOV
# after 1 and 2 global rotations, per-action lookahead vs those unions, and enemy slots.
FOV_DISTANCE_PRED = 4
MAX_ENEMIES_PRED = 8
_OBS5_EXTRA_HEAD = 5  # seen_now, in_next, in_2, n_enemies_norm, bfs_dist_norm
_OBS5_EXTRA_ACTION = 10  # 5 actions × (safe @ +1 rot, safe @ +2 rot)
_OBS5_SLOT = 7  # row, col, orient one-hot×4, dist_to agent
OBS5_DIM = OBS3_DIM + _OBS5_EXTRA_HEAD + _OBS5_EXTRA_ACTION + MAX_ENEMIES_PRED * _OBS5_SLOT


def _simulated_fov_cells(
    ey: int,
    ex: int,
    orientation: int,
    walls: set,
    enemy_bodies: set,
    rows: int,
    cols: int,
    fov_distance: int = FOV_DISTANCE_PRED,
) -> set[tuple[int, int]]:
    """Ray-cast FOV from one enemy; walls and any enemy cell block (matches env visibility)."""
    out = set()
    if orientation == 0:  # LEFT — col decreases
        dr, dc = 0, -1
    elif orientation == 1:  # DOWN — row increases
        dr, dc = 1, 0
    elif orientation == 2:  # RIGHT
        dr, dc = 0, 1
    else:  # UP
        dr, dc = -1, 0
    for i in range(1, fov_distance + 1):
        r, c = ey + dr * i, ex + dc * i
        if r < 0 or r >= rows or c < 0 or c >= cols:
            break
        if (r, c) in walls or (r, c) in enemy_bodies:
            break
        out.add((r, c))
    return out


def _infer_enemy_orientations(
    grid: np.ndarray,
    enemy_cells: list[tuple[int, int]],
    walls: set,
    enemy_bodies: set,
    rows: int,
    cols: int,
) -> list[int]:
    """Pick orientation per enemy by best overlap with RED/LIGHT_RED cells on the simulated ray."""
    danger_cells = set()
    for r in range(rows):
        for c in range(cols):
            if cell_type(grid[r, c]) in (5, 6):
                danger_cells.add((r, c))
    orientations = []
    for ey, ex in enemy_cells:
        best_o, best_score = 0, -1
        for o in range(4):
            sim = _simulated_fov_cells(ey, ex, o, walls, enemy_bodies, rows, cols)
            score = len(sim & danger_cells)
            if score > best_score:
                best_score = score
                best_o = o
        orientations.append(best_o)
    return orientations


def _union_predicted_fov(
    enemy_cells: list[tuple[int, int]],
    orientations: list[int],
    orient_offset: int,
    walls: set,
    enemy_bodies: set,
    rows: int,
    cols: int,
) -> set[tuple[int, int]]:
    u = set()
    for (ey, ex), base_o in zip(enemy_cells, orientations):
        o = (base_o + orient_offset) % 4
        u |= _simulated_fov_cells(ey, ex, o, walls, enemy_bodies, rows, cols)
    return u


def obs_space_enemy_predictive(env: gym.Env) -> gym.spaces.Space:
    return gym.spaces.Box(low=0.0, high=1.0, shape=(OBS5_DIM,), dtype=np.float32)


def obs_enemy_predictive(grid: np.ndarray) -> np.ndarray:
    """
    Mode 3 prefix (obs_local_patch, OBS3_DIM) plus predictive suffix only:
      - agent_seen_now, agent_in_predicted_FOV (+1 / +2 rotations), n_enemies_norm, bfs_dist_norm
      - per action (L,D,R,U,stay): safe landing vs inferred FOV after +1 and +2 rotations
      - up to 8 enemy slots (sorted by row, col): position, orientation one-hot, dist to agent
    """
    base = obs_local_patch(grid)
    assert base.shape[0] == OBS3_DIM

    metrics = _grid_metrics(grid)
    rows, cols = metrics["rows"], metrics["cols"]
    agent_r, agent_c = metrics["agent_r"], metrics["agent_c"]
    max_manhattan = metrics["max_manhattan"]

    walls = set(_WALL_CELLS)
    enemy_cells = sorted(
        (r, c) for r in range(rows) for c in range(cols) if cell_type(grid[r, c]) == 4
    )
    enemy_bodies = set(enemy_cells)

    bfs_dist, _, _ = _bfs_nearest_unexplored(grid, agent_r, agent_c)
    bfs_dist_norm = (
        min(float(bfs_dist) / float(max(rows + cols - 2, 1)), 1.0) if bfs_dist > 0 else 0.0
    )

    agent_seen_now = 1.0 if cell_type(grid[agent_r, agent_c]) in (5, 6) else 0.0

    orientations = _infer_enemy_orientations(grid, enemy_cells, walls, enemy_bodies, rows, cols)
    fov_next = _union_predicted_fov(enemy_cells, orientations, 1, walls, enemy_bodies, rows, cols)
    fov_2 = _union_predicted_fov(enemy_cells, orientations, 2, walls, enemy_bodies, rows, cols)

    agent_in_next = 1.0 if (agent_r, agent_c) in fov_next else 0.0
    agent_in_2 = 1.0 if (agent_r, agent_c) in fov_2 else 0.0

    action_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    action_bits = []
    for dr, dc in action_deltas:
        nr, nc = agent_r + dr, agent_c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls or (nr, nc) in enemy_bodies:
            dest_r, dest_c = agent_r, agent_c
        else:
            dest_r, dest_c = nr, nc
        action_bits.append(0.0 if (dest_r, dest_c) in fov_next else 1.0)
        action_bits.append(0.0 if (dest_r, dest_c) in fov_2 else 1.0)

    n_enemies_norm = min(len(enemy_cells), MAX_ENEMIES_PRED) / float(MAX_ENEMIES_PRED)

    slot_feats = []
    for i in range(MAX_ENEMIES_PRED):
        if i < len(enemy_cells):
            ey, ex = enemy_cells[i]
            o = orientations[i]
            oh = [0.0] * 4
            oh[o] = 1.0
            dist = abs(agent_r - ey) + abs(agent_c - ex)
            slot_feats.extend(
                [
                    ey / max(rows - 1, 1),
                    ex / max(cols - 1, 1),
                    oh[0],
                    oh[1],
                    oh[2],
                    oh[3],
                    dist / max_manhattan,
                ]
            )
        else:
            slot_feats.extend([0.0] * _OBS5_SLOT)

    head = [
        agent_seen_now,
        agent_in_next,
        agent_in_2,
        n_enemies_norm,
        bfs_dist_norm,
    ]
    assert len(head) == _OBS5_EXTRA_HEAD
    assert len(action_bits) == _OBS5_EXTRA_ACTION

    tail = np.asarray(head + action_bits + slot_feats, dtype=np.float32)
    out = np.concatenate([base, tail])
    assert out.shape[0] == OBS5_DIM
    return out


# PUBLIC API  (called by env.py)
 
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Returns the observation space based on OBS_MODE.
 
    OBS_MODE 1 → MultiDiscrete([7]*100)   — full grid encoding
    OBS_MODE 2 → Box(3,)                  — compact feature vector
    OBS_MODE 3 → Box(59,)                 — local patch + global metrics + action flags
    OBS_MODE 4 → Box(12,)                 — frontier/risk features
    OBS_MODE 5 → Box(130,)                 — obs_local_patch (59) + predictive FOV tail (71)
    """
    if OBS_MODE == 1:
        return obs_space_full(env)
    if OBS_MODE == 2:
        return obs_space_compact(env)
    if OBS_MODE == 3:
        return obs_space_local_patch(env)
    if OBS_MODE == 4:
        return obs_space_frontier_risk(env)
    if OBS_MODE == 5:
        return obs_space_enemy_predictive(env)
    raise ValueError("OBS_MODE must be 1, 2, 3, 4, or 5")
 
 
def observation(grid: np.ndarray) -> np.ndarray:
    """
    Returns the observation for the current grid state.
    Must match the shape/dtype declared in observation_space().
    """
    global _WALL_CELLS, _CURR_NEAREST_UNEXPLORED_DIST, _GRID_SNAPSHOT_FOR_REWARD
    rows, cols = grid.shape[:2]

    _GRID_SNAPSHOT_FOR_REWARD = np.array(grid, copy=True)

    # Single pass: cache walls, agent position, and nearest unexplored distance
    # so both the obs and reward() functions can use them without re-scanning.
    _WALL_CELLS = set()
    agent_r, agent_c = 0, 0
    unexplored = []
    for r in range(rows):
        for c in range(cols):
            ct = cell_type(grid[r, c])
            if ct == 2:
                _WALL_CELLS.add((r, c))
            elif ct == 3:
                agent_r, agent_c = r, c
            elif ct in (0, 5):
                unexplored.append((r, c))

    if unexplored:
        _CURR_NEAREST_UNEXPLORED_DIST = float(
            min(abs(agent_r - r) + abs(agent_c - c) for r, c in unexplored)
        )
    else:
        _CURR_NEAREST_UNEXPLORED_DIST = 0.0

    if OBS_MODE == 1:
        return obs_full(grid)
    if OBS_MODE == 2:
        return obs_compact(grid)
    if OBS_MODE == 3:
        return obs_local_patch(grid)
    if OBS_MODE == 4:
        return obs_frontier_risk(grid)
    if OBS_MODE == 5:
        return obs_enemy_predictive(grid)
    raise ValueError("OBS_MODE must be 1, 2, 3, 4, or 5")



def _flat_to_row_col(pos: int, grid_size: int = 10) -> tuple[int, int]:
    return pos // grid_size, pos % grid_size


def _predict_next_enemy_fov(
    enemy,
    grid_size: int = 10,
    fov_distance: int = 4,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    return _predict_enemy_fov_after_k_steps(
        enemy,
        steps_ahead=1,
        grid_size=grid_size,
        fov_distance=fov_distance,
        blocked_cells=blocked_cells,
    )


def _predict_enemy_fov_after_k_steps(
    enemy,
    steps_ahead: int,
    grid_size: int = 10,
    fov_distance: int = 4,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    orientation = (enemy.orientation + steps_ahead) % 4
    out = set()
    blocked = blocked_cells if blocked_cells is not None else set()
    for i in range(1, fov_distance + 1):
        if orientation == 0:  # LEFT
            rr, cc = enemy.y, enemy.x - i
        elif orientation == 1:  # DOWN
            rr, cc = enemy.y + i, enemy.x
        elif orientation == 2:  # RIGHT
            rr, cc = enemy.y, enemy.x + i
        else:  # UP
            rr, cc = enemy.y - i, enemy.x
        if 0 <= rr < grid_size and 0 <= cc < grid_size:
            # Mirror env __is_cell_visible() partially:
            # stop ray if the candidate cell is occupied by an enemy.
            # (wall blocking is unknown in reward() because grid is not provided in info.)
            if (rr, cc) in blocked:
                break
            out.add((rr, cc))
        else:
            break
    return out


def _candidate_cells(
    cell: tuple[int, int],
    grid_size: int = 10,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return local action candidates: stay + 4-neighborhood."""
    r, c = cell
    blocked = blocked_cells if blocked_cells is not None else set()
    out = [(r, c)]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < grid_size and 0 <= cc < grid_size and (rr, cc) not in blocked:
            out.append((rr, cc))
    return out


def _cell_death_rank(
    cell: tuple[int, int],
    fov_1: set,
    fov_2: set,
    grid_size: int = 10,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> int:
    """
    Death-only ranking (lower is better):
      0: not guaranteed dead in 2 steps
      1: guaranteed dead in 2 steps
      2: guaranteed dead in 1 step
    """
    # 1) Dead immediately:
    # stepping into this cell means the agent is seen on the very next enemy update.
    if cell in fov_1:
        return 2

    # 2) Dead soon:
    # if stepping here, all local choices in the following turn are seen in 2 steps.
    one_step_choices_after_entry = _candidate_cells(cell, grid_size, blocked_cells=blocked_cells)
    dead_in_2 = all(candidate in fov_2 for candidate in one_step_choices_after_entry)
    if dead_in_2:
        return 1
    return 0


def _nearest_manhattan_distance(src: tuple[int, int], targets: list[tuple[int, int]], default_value: float = 20.0) -> float:
    if not targets:
        return default_value
    sr, sc = src
    return float(min(abs(sr - tr) + abs(sc - tc) for (tr, tc) in targets))


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not

    Which reward branch runs is controlled by the module-level REWARD_MODE (1–4), like OBS_MODE for observations.
    """
    reward_func = REWARD_MODE
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    global _LAST_AGENT_POS, _PREV_AGENT_POS, _STILL_COUNT, _TWO_STEP_LOOP_COUNT, _LAST_STEPS_REMAINING, _PREV_SAFE_FRONTIER_DIST, _CURR_NEAREST_UNEXPLORED_DIST, _VISITED_POSITIONS, _NO_PROGRESS_STREAK, _WALL_CELLS, _RECENT_POSITIONS, _STEPS_SINCE_DISCOVERY, _NEXT_FOV_CELLS, _PREV_BFS_DIST, _REWARD4_EPISODE_VISITED

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    final_reward = 0

    if reward_func == 1:
        if new_cell_covered:
            final_reward += 1

        final_reward += (total_covered_cells/coverable_cells) 
    elif reward_func == 2:
        if not game_over:
            final_reward += 10*len(enemies)

        final_reward += total_covered_cells
    elif reward_func == 3:
        if new_cell_covered: #rewards new discovery, increasing with even more enemies
            final_reward += 1*len(enemies)
        
        final_reward += (total_covered_cells - steps_remaining) + agent_pos
    elif reward_func == 4:
        # Enhanced reward function 4:
        # - Smart danger avoidance with lookahead
        # - Strong exploration encouragement
        # - Anti-loop mechanisms
        # - Progressive difficulty scaling
        
        agent_rc = _flat_to_row_col(agent_pos)
        has_enemies = len(enemies) > 0
        coverage_ratio = total_covered_cells / max(coverable_cells, 1)

        if has_enemies:
            enemy_cells = {(enemy.y, enemy.x) for enemy in enemies}
            blocked_cells = set(enemy_cells) | set(_WALL_CELLS)

            current_fov = set()
            next_fov = set()
            fov_2_steps = set()

            for enemy in enemies:
                current_fov.update(enemy.get_fov_cells())
                next_fov.update(_predict_next_enemy_fov(enemy, blocked_cells=blocked_cells))
                fov_2_steps.update(
                    _predict_enemy_fov_after_k_steps(enemy, 2, blocked_cells=blocked_cells)
                )

            # Cache 2-step-ahead FOV for obs_local_patch. Because observation() is called
            # before reward() in each env step, the obs at step t+1 reads this value set
            # at step t — making 2-ahead here equal to 1-ahead from the obs perspective.
            _NEXT_FOV_CELLS = fov_2_steps

            immediate_danger = agent_rc in next_fov
            currently_seen = agent_rc in current_fov
            unseen_now = not currently_seen
        else:
            _NEXT_FOV_CELLS = set()
            immediate_danger = False
            currently_seen = False
            unseen_now = True

        # Dull first-visit / frontier / BFS+ when in cone or about to enter (choke timing)
        unsafe_explore = has_enemies and (immediate_danger or currently_seen)
        explore_scale = REWARD4_UNSAFE_EXPLORE_SCALE if unsafe_explore else 1.0
        if (
            unsafe_explore
            and has_enemies
            and len(enemies) >= REWARD4_MANY_ENEMIES_THRESHOLD
        ):
            explore_scale = max(explore_scale, REWARD4_UNSAFE_EXPLORE_SCALE_DENSE)
        # Withhold big +new-cell only when already *seen* (in FOV). Using immediate_danger here
        # starves sneaky_enemies: next_fov is true on many safe approach tiles.
        unsafe_newcell = has_enemies and currently_seen

        
        # Episode boundary detection: reset anti-loop trackers when a new episode starts.
        if _LAST_STEPS_REMAINING is None or steps_remaining >= _LAST_STEPS_REMAINING:
            _RECENT_POSITIONS.clear()
            _STEPS_SINCE_DISCOVERY = 0
            _PREV_BFS_DIST = None
            _REWARD4_EPISODE_VISITED = set()
        _RECENT_POSITIONS.append(agent_pos)
        if len(_RECENT_POSITIONS) > _RECENT_WINDOW:
            _RECENT_POSITIONS.pop(0)
        _LAST_STEPS_REMAINING = steps_remaining

        # --- Exploration reward ---
        # Blunt the +5 / low-coverage / unseen spike when standing on a tile the next
        # enemy sweep would hit (immediate_danger). Otherwise new-cell return ~10/step can
        # dominate sparse death (-88) in the critic.
        newcell_scale = (
            REWARD4_NEWCELL_DANGER_SCALE
            if has_enemies and immediate_danger
            else 1.0
        )
        if new_cell_covered:
            _STEPS_SINCE_DISCOVERY = 0
            # No +5 / low-coverage spike only when standing in lit FOV (not mere "next step" risk)
            if not unsafe_newcell:
                final_reward += 5.0 * newcell_scale
                if coverage_ratio < REWARD4_LOW_COVERAGE_THRESHOLD:
                    final_reward += REWARD4_LOW_COVERAGE_NEWCELL_EXTRA * newcell_scale
            if unseen_now and enemies and not unsafe_newcell:
                final_reward += 1.5 * newcell_scale
        else:
            _STEPS_SINCE_DISCOVERY += 1
            # Penalty escalates based on how long since the last new cell,
            # NOT total steps elapsed. This catches arbitrarily long loops —
            # including wall-corridor back-and-forth that exceed our fixed
            # cycle-detection window.
            drought = _STEPS_SINCE_DISCOVERY
            if drought < 20:
                final_reward -= 1.0
            elif drought < 50:
                final_reward -= 1.5
            elif drought < 100:
                final_reward -= 2.5
            else:
                final_reward -= 4.0   # extreme pressure: must find a new cell


        # --- Enemy danger (disabled when no enemies: just_go / safe navigation) ---
        if has_enemies:
            if immediate_danger:
                final_reward -= REWARD4_IMMEDIATE_DANGER_PENALTY
            if currently_seen:
                final_reward -= REWARD4_CURRENTLY_SEEN_PENALTY
            elif not immediate_danger:
                # No +1 if next enemy rotation would catch us — avoids rewarding "hover at cone edge"
                final_reward += 1.0

        # --- Path diversity: reward each distinct cell position at most once per episode ---
        if agent_pos not in _REWARD4_EPISODE_VISITED:
            _REWARD4_EPISODE_VISITED.add(agent_pos)
            final_reward += REWARD4_FIRST_VISIT_BONUS * explore_scale

        # --- Frontier hugging: small bonus when standing next to still-unexplored tiles ---
        g = _GRID_SNAPSHOT_FOR_REWARD
        if g is not None and cells_remaining > 0:
            if _agent_adjacent_unexplored(g, agent_rc[0], agent_rc[1]):
                final_reward += REWARD4_FRONTIER_ADJ_BONUS * explore_scale

        # --- Walkable progress toward nearest unexplored (BFS), clipped ---
        if g is not None:
            bfs_dist, _, _ = _bfs_nearest_unexplored(g, agent_rc[0], agent_rc[1])
            curr_bfs = float(bfs_dist)
            bfs_scale = 1.0
            if coverage_ratio < REWARD4_EXPLORE_BOOST_COVER_MAX:
                bfs_scale = 1.0 + REWARD4_EXPLORE_BOOST_BFS * (
                    (REWARD4_EXPLORE_BOOST_COVER_MAX - coverage_ratio)
                    / max(REWARD4_EXPLORE_BOOST_COVER_MAX, 1e-6)
                )
            if _PREV_BFS_DIST is not None:
                delta = _PREV_BFS_DIST - curr_bfs
                shaped = REWARD4_BFS_ALPHA * bfs_scale * delta
                # Under enemy gaze / next-step FOV: dull "push toward frontier", keep full retreat signal
                if unsafe_explore and delta > 0:
                    shaped *= explore_scale
                if shaped > REWARD4_BFS_SHAPING_CAP:
                    shaped = REWARD4_BFS_SHAPING_CAP
                elif shaped < -REWARD4_BFS_SHAPING_CAP:
                    shaped = -REWARD4_BFS_SHAPING_CAP
                final_reward += shaped
            _PREV_BFS_DIST = curr_bfs

        # --- Cycle penalty: catches N-cycles for N = 2, 3, 4, 5, 6, 7 ---
        # Previous code only caught 2-cycles and 3-cycles; 4+ cycles were a
        # free exploit — agent ran a deterministic 4-cycle in eval and never
        # broke out of it (no stochastic noise in deterministic=True mode).
        short_loop  = len(_RECENT_POSITIONS) >= 4  and len(set(_RECENT_POSITIONS[-4:]))  <= 2
        medium_loop = len(_RECENT_POSITIONS) >= 6  and len(set(_RECENT_POSITIONS[-6:]))  <= 3
        long_loop   = len(_RECENT_POSITIONS) >= 8  and len(set(_RECENT_POSITIONS[-8:]))  <= 4
        xlong_loop  = len(_RECENT_POSITIONS) >= 10 and len(set(_RECENT_POSITIONS[-10:])) <= 5
        xxlong_loop = len(_RECENT_POSITIONS) >= 12 and len(set(_RECENT_POSITIONS[-12:])) <= 6
        xxxlong_loop= len(_RECENT_POSITIONS) >= 14 and len(set(_RECENT_POSITIONS[-14:])) <= 7
        is_looping  = short_loop or medium_loop or long_loop or xlong_loop or xxlong_loop or xxxlong_loop

        if is_looping:
            final_reward -= 5.0

        # --- Victory (unclipped) + clip dense shaping so step returns do not explode ---
        victory_bonus = 200.0 if cells_remaining == 0 else 0.0
        final_reward = max(
            REWARD4_STEP_CLIP_MIN, min(REWARD4_STEP_CLIP_MAX, final_reward)
        )
        final_reward += victory_bonus
    else:
        raise ValueError("REWARD_MODE must be 1 (balanced), 2 (safe), 3 (greedy), or 4 (enhanced)")
    
    if game_over:
        if reward_func == 4:
            coverage_ratio_term = total_covered_cells / max(coverable_cells, 1)
            final_reward = (REWARD4_GAME_OVER_PENALTY
                            + REWARD4_TERMINAL_COVERAGE_BONUS * coverage_ratio_term)
        else:
            final_reward = -60
    return final_reward
