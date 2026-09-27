import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip
from mcts_agent import MCTSAgent

def main():
    # Create the CartPole environment
    env = gym.make('CartPole-v1', render_mode=None)

    # Wrap the environment to record videos 
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda episode_id: True  )# record every episode)
    # Initialize the MCTS agent
    agent = MCTSAgent('CartPole-v1', explore_iterations=5, c=1.0)

    # Number of episodes to run
    num_episodes = 10
    max_steps = 50

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\nEpisode {episode + 1}:")

        while not done and step < max_steps: 
            # Get action from MCTS agent
            action = agent.get_action(env, observation, done)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step += 1

        print(f"Total Reward: {total_reward}, Steps: {step}")

        # Reset MCTS tree between episodes
        agent.reset()

    env.close()

    # capture the video and convert to gif
    clip = VideoFileClip("videos/rl-video-episode-0.mp4")
    clip.write_gif("cartpole.gif")


if __name__ == "__main__":
    main()