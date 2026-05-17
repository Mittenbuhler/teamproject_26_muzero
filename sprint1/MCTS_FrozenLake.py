import gymnasium as gym
from mcts_agent import MCTSAgent

class HolePenaltyWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward == 0.0:
            reward = -1.0
        return observation, reward, terminated, truncated, info
class DistanceRewardWrapper(gym.Wrapper):
    def __init__(self, env, distance_scale=1.0, wall_penalty=-1.0, reversal_penalty=-2.0):
        super().__init__(env)
        self.ncol = env.unwrapped.desc.shape[1]
        self.goal_state = self._find_goal_state()
        self.distance_scale = distance_scale
        self.wall_penalty = wall_penalty
        self.reversal_penalty = reversal_penalty
        self.prev_obs = None
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Distance shaping
        distance_bonus = self.distance_scale * self._distance_shaping(self.prev_obs, obs)
        reward += distance_bonus
        
        # Wall penalty
        if not terminated and not truncated and obs == self.prev_obs:
            reward += self.wall_penalty
        
        # Reversal penalty
        if self.prev_action is not None and self._is_opposite_action(self.prev_action, action):
            reward += self.reversal_penalty
        
        self.prev_obs = obs
        self.prev_action = action
        return obs, reward, terminated, truncated, info

    def _is_opposite_action(self, prev_action, current_action):
        opposites = {0: 2, 2: 0, 1: 3, 3: 1}  # LEFT-RIGHT, DOWN-UP
        return opposites.get(prev_action) == current_action

    def _coords(self, obs):
        return obs // self.ncol, obs % self.ncol

    def _distance(self, obs):
        r, c = self._coords(obs)
        gr, gc = self._coords(self.goal_state)
        return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5

    def _distance_shaping(self, prev_obs, obs):
        return self._distance(prev_obs) - self._distance(obs)

    def _find_goal_state(self):
        desc = self.env.unwrapped.desc
        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == b'G':
                    return i * self.ncol + j
        raise ValueError("Goal not found")
    
def main():
    # Create the FrozenLake environment (deterministic for easier learning)
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
    env = HolePenaltyWrapper(env)
    env = DistanceRewardWrapper(env)

    # Initialize the MCTS agent
    agent = MCTSAgent(env, explore_iterations=2000, c=0.7)

    # Number of episodes to run
    num_episodes = 10

    for episode in range(num_episodes):
        # Reset the environment
        observation, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\nEpisode {episode + 1}:")

        while not done and step < 100:  # Limit steps to prevent infinite loops
            # Get action from MCTS agent
            action = agent.get_action(env, observation, done)

            # Take the action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Render the environment as ANSI and print it
            print(env.render())

        print(f"Total Reward: {total_reward}, Steps: {step}")

        # Reset the agent's tree for the next episode
        agent.reset()

    env.close()

if __name__ == "__main__":
    main()