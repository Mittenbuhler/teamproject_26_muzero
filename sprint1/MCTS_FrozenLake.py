import gymnasium as gym
import imageio
from sprint1.mcts_agent import MCTSAgent

def main():
    # Create the FrozenLake environment (deterministic for easier learning)
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')

    # Initialize the MCTS agent
    agent = MCTSAgent('FrozenLake-v1', explore_iterations=2000, c=1.0)

    # Number of episodes to run
    num_episodes = 10

    frames = []  # To store frames for GIF

    for episode in range(num_episodes):
        # Reset the environment
        observation, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\nEpisode {episode + 1}:")

        while not done and step < 100:  # Limit steps to prevent infinite loops
            frames.append(env.render()) # Store the current frame for GIF creation

            # Get action from MCTS agent
            action = agent.get_action(env, observation, done)

            # Take the action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Optional: Render the environment (prints the grid)
            print(env.render())

        print(f"Total Reward: {total_reward}, Steps: {step}")

        # Reset the agent's tree for the next episode
        agent.reset()

    env.close()

    imageio.mimsave("frozenlake.gif", frames, fps=2)  # Save frames as a GIF with 2 frames per second

if __name__ == "__main__":
    main()