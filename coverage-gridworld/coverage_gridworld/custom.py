import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
    # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).
    cell_values = env.grid + 256

    # if MultiDiscrete is used, it's important to flatten() numpy arrays!
    return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # If the observation returned is not the same shape as the observation_space, an error will occur!
    # Make sure to make changes to both functions accordingly.

    return grid.flatten()


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
            final_reward += 10*enemies

        final_reward += cells_remaining
    elif reward_func == 3:
        if new_cell_covered: #rewards new discovery, increasing with even more enemies
            final_reward += 1*enemies
        
        final_reward += (total_covered_cells - steps_remaining) + agent_pos
    else:
        raise ValueError("reward_func only accepts values 1 (balanced), 2 (safe) or 3 (greedy)")
    
    if game_over:
        final_reward = -100
    return final_reward
