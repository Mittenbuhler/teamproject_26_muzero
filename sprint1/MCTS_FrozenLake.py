from math import ceil

import gymnasium as gym
from mcts_agent import MCTSAgent

class HoleAndGoalWrapper(gym.Wrapper):
    def __init__(self, env, hole_penalty=-5.0, goal_reward=10.0):
        super().__init__(env)
        self.hole_states = self._find_hole_states()
        self.goal_state = self._find_goal_state()
        self.hole_penalty = hole_penalty
        self.goal_reward = goal_reward

    def _find_hole_states(self):
        """Find all hole positions in the FrozenLake grid."""
        holes = []
        desc = self.env.unwrapped.desc
        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == b'H':
                    holes.append(i * len(row) + j)
        return holes

    def _find_goal_state(self):
        """Find the goal position."""
        desc = self.env.unwrapped.desc
        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == b'G':
                    return i * len(row) + j
        return None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation in self.hole_states:
            reward = self.hole_penalty
        elif observation == self.goal_state:
            reward = self.goal_reward
        return observation, reward, terminated, truncated, info
class DistanceRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        distance_scale=1.0,
        wall_penalty=-1.0,
        reversal_penalty=0,
    ):
        super().__init__(env)

        self.ncol = env.unwrapped.desc.shape[1]
        self.goal_state = self._find_goal_state()
        self.hole_states = self._find_hole_states()

        self.distance_scale = distance_scale
        self.wall_penalty = wall_penalty
        self.reversal_penalty = reversal_penalty

        self.prev_obs = None
        self.prev_action = None

        # NEW: consecutive progress counter
        self.progress_streak = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_obs = obs
        self.prev_action = None
        self.progress_streak = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        moved = obs != self.prev_obs
        reversed_dir = (
            self.prev_action is not None
            and self._is_opposite_action(self.prev_action, action)
        )

        # Distance improvement
        distance_delta = self._distance_shaping(self.prev_obs, obs)

        # Reward shaping
        if (
            moved
            and not reversed_dir
            and obs not in self.hole_states
            and distance_delta > 0
        ):
            # Increase streak
            self.progress_streak += 1

            # Doubles every consecutive successful step
            multiplier = 2 ** (self.progress_streak - 1)

            distance_bonus = (
                self.distance_scale
                * distance_delta
                * multiplier
            )

            reward += distance_bonus

        else:
            # Reset streak if no progress
            self.progress_streak = 0

        # Wall penalty
        if not terminated and not truncated and not moved:
            reward += self.wall_penalty

        # Reversal penalty
        if (
            self.prev_action is not None
            and moved
            and reversed_dir
        ):
            reward += self.reversal_penalty

        self.prev_obs = obs
        self.prev_action = action

        return obs, reward, terminated, truncated, info

    def _is_opposite_action(self, prev_action, current_action):
        opposites = {
            0: 2,  # LEFT -> RIGHT
            2: 0,  # RIGHT -> LEFT
            1: 3,  # DOWN -> UP
            3: 1,  # UP -> DOWN
        }
        return opposites.get(prev_action) == current_action

    def _coords(self, obs):
        return obs // self.ncol, obs % self.ncol

    def _distance(self, obs):
        r, c = self._coords(obs)
        gr, gc = self._coords(self.goal_state)

        return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5

    def _distance_shaping(self, prev_obs, obs):
        return ceil(self._distance(prev_obs) - self._distance(obs))

    def _find_hole_states(self):
        desc = self.env.unwrapped.desc

        holes = []

        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == b'H':
                    holes.append(i * self.ncol + j)

        return holes

    def _find_goal_state(self):
        desc = self.env.unwrapped.desc

        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == b'G':
                    return i * self.ncol + j

        raise ValueError("Goal not found")

def create_frozenlake_env():
    """Factory function that creates a fresh FrozenLake environment with all wrappers applied."""
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
    env = HoleAndGoalWrapper(env)
    env = DistanceRewardWrapper(env)
    return env

def main():
    # Create the initial environment for the main loop
    env = create_frozenlake_env()

    # Initialize the MCTS agent with the factory function
    agent = MCTSAgent(env_factory=create_frozenlake_env, explore_iterations=5000, c=0.7)

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
            print(f"Step {step}: reward={reward}")

        print(f"Total Reward: {total_reward}, Steps: {step}")

        # Reset the agent's tree for the next episode
        agent.reset()

    env.close()

if __name__ == "__main__":
    main()