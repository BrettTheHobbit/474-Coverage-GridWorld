import gymnasium
import coverage_gridworld

env = gymnasium.make("just_go", render_mode=None)
obs, info = env.reset()
print("Start obs[0:5]:", obs[:5])  # first 5 cells

for action in [2, 2, 2]:  # move right 3 times
    obs, reward, done, truncated, info = env.step(action)
    print(f"action={action} | obs[0:5]: {obs[:5]}")

env.close()
