import flappy_bird_gymnasium
import gymnasium
from PIL import Image as im
import numpy as np

env = gymnasium.make(
    "FlappyBird-v0", render_mode="human", use_pixels=True, training_mode=True
)

obs, _ = env.reset()

idx = 0
total_reward = 0.0

while True:
    # Next action:
    # (feed the observation to your agent here)
    # action = env.action_space.sample()
    action = 1

    # Processing:
    obs, reward, terminated, _, info = env.step(action)

    # print("reward=", reward, type(reward))
    total_reward += reward
    # print("Reward=", total_reward)

    image = obs
    # print("Shape is:", obs.shape)
    grayscale_image = (
        0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    )

    normalized_grayscale_image = grayscale_image / 255.0
    # print(normalized_grayscale_image)

    # print(normalized_grayscale_image.shape)
    grayscale_image_uint8 = (grayscale_image * 255).astype(np.uint8)
    # print(normalized_grayscale_image)

    # print(env.observation_space)
    # print(obs)
    data = im.fromarray(grayscale_image_uint8)
    # saving the final output
    # as a PNG file
    data.save(f"output/gfg_dummy_pic_{str(idx).rjust(3, "0")}.png")
    idx += 1

    # Checking if the player is still alive
    if terminated:
        break

env.close()

print("Final reward=", total_reward)
