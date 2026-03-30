import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

#Choosing Observation Mode
OBS_MODE = 2   # 1 = full grid  |  2 = compact feature vector
 
# Color constants 
_BLACK      = (0,   0,   0)    # unexplored cell
_WHITE      = (255, 255, 255)  # explored cell
_BROWN      = (101, 67,  33)   # wall
_GREY       = (160, 161, 161)  # agent
_GREEN      = (31,  198, 0)    # enemy
_RED        = (255, 0,   0)    # unexplored cell under enemy FOV
_LIGHT_RED  = (255, 127, 127)  # explored cell under enemy FOV
 

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


# PUBLIC API  (called by env.py)
 
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Returns the observation space based on OBS_MODE.
 
    OBS_MODE 1 → MultiDiscrete([7]*100)  — full grid encoding
    OBS_MODE 2 → Box(3,)                 — compact feature vector  ← default
    """
    if OBS_MODE == 1:
        return obs_space_full(env)
    else:
        return obs_space_compact(env)
 
 
def observation(grid: np.ndarray) -> np.ndarray:
    """
    Returns the observation for the current grid state.
    Must match the shape/dtype declared in observation_space().
    """
    if OBS_MODE == 1:
        return obs_full(grid)
    else:
        return obs_compact(grid)



def reward(info: dict, reward_func: int) -> float:
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
    
    reward_func - 1 2 or 3 and determines which reward calculation is used
        1 - A balanced reward that takes into account most information equally
        2 - A conservative function, prioritizing safety above all else
        3 - A greedier function, prioritizing covering as many cells as possible without endangering itself
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    final_reward = 0

    if reward_func == 1:
        if new_cell_covered:
            final_reward += 1

        final_reward = (total_covered_cells/coverable_cells) - (cells_remaining/coverable_cells)
    elif reward_func == 2:
        if not game_over:
            final_reward += 10*len(enemies)

        final_reward += cells_remaining
    elif reward_func == 3:
        if new_cell_covered: #rewards new discovery, increasing with even more enemies
            final_reward += 1*len(enemies)
        
        final_reward += (total_covered_cells - steps_remaining) + agent_pos
    else:
        raise ValueError("reward_func only accepts values 1 (balanced), 2 (safe) or 3 (greedy)")
    
    if game_over:
        final_reward = -100
    return final_reward
