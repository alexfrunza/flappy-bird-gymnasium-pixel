import flappy_bird_gymnasium
import gymnasium
from PIL import Image as im

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_pixels=True)

obs, _ = env.reset()

idx = 0

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    print(obs)
    data = im.fromarray(obs)
    # saving the final output
    # as a PNG file
    data.save(f"output/gfg_dummy_pic_{str(idx).rjust(3, "0")}.png")
    idx += 1

    # Checking if the player is still alive
    if terminated:
        break

env.close()
