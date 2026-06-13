import numpy as np
import gymnasium as gym
import random
import torch
from math import sqrt, log

# MCTS Node class represents a node of the tree and contains information needed for
# the algorithm to run its search.
class MCTSNode:
    def __init__(
        self,
        game,
        done,
        parent,
        observation,
        action_index,
        game_name,
        c=1.0,
        env_factory=None,
        dynamics_model=None,
        policy_network=None,
        value_network=None,
        observation_encoder=None,
        prior_probability=None,
        reward=0.0,
        terminal_fn=None,
        discount=1.0
    ):

        # child nodes
        self.child = None

        # total rewards from MCTS exploration
        self.T = 0

        # visit count
        self.N = 0

        # the game environment
        self.game = game

        # observation of the environment
        self.observation = observation

        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent

        # action index that led to this node
        self.action_index = action_index

        # immediate reward received when entering this node
        self.reward = reward

        # game name
        self.game_name = game_name

        # exploration constant
        self.c = c

        # environment factory for creating wrapped environments
        self.env_factory = env_factory

        # optional learned model used for expansion
        self.dynamics_model = dynamics_model
        self.terminal_fn = terminal_fn
        self.discount = discount

        # optional policy/value networks used by AlphaZero-style search
        self.policy_network = policy_network
        self.value_network = value_network
        self.observation_encoder = observation_encoder
        self.prior_probability = prior_probability

        # get the info from the game environment
        if game is not None:
            self.action_space = game.action_space.n
            if len(game.observation_space.shape) > 0:
                self.game_obs = game.observation_space.shape[0]
            else:
                self.game_obs = game.observation_space.n
        elif dynamics_model is not None:
            self.action_space = dynamics_model.action_dim

        if self.prior_probability is None:
            self.prior_probability = 1.0 / self.action_space

    # getUCBscore is the formula that gives a value to the node.
    # MCTS will pick the nodes with the highest value.        
    def getUCBscore(self):

        if self.policy_network is not None and self.parent is not None:
            q_value = 0.0 if self.N == 0 else self.T / self.N
            parent_visits = max(self.parent.N, 1)
            prior_score = (
                self.c
                * self.prior_probability
                * sqrt(parent_visits)
                / (1 + self.N)
            )
            return q_value + prior_score

        # Unexplored nodes get a max value to favour exploration
        if self.N == 0:
            return float('inf')
        
        # Get information about the parent node of current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        
        # Use one of the possible MCTS formula for calculating the node value 
        return (self.T / self.N) + self.c * sqrt(log(top_node.N) / self.N)

    def _network_input_size(self, network):
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                return module.in_features
        return None

    def encode_observation(self, observation, network):
        if self.observation_encoder is not None:
            encoded = self.observation_encoder(observation)
        else:
            encoded = observation

        if isinstance(encoded, torch.Tensor):
            state = encoded.detach().float()
        else:
            state = torch.as_tensor(encoded, dtype=torch.float32)

        state = state.flatten()
        input_size = self._network_input_size(network)

        if input_size is not None:
            if state.numel() < input_size:
                state = torch.nn.functional.pad(state, (0, input_size - state.numel()))
            elif state.numel() > input_size:
                state = state[:input_size]

        return state.unsqueeze(0).to(next(network.parameters()).device)

    def get_policy_priors(self):
        if self.policy_network is None:
            return np.ones(self.action_space, dtype=np.float32) / self.action_space

        was_training = self.policy_network.training
        self.policy_network.eval()

        with torch.no_grad():
            state = self.encode_observation(self.observation, self.policy_network)
            policy_output = self.policy_network(state)

            if isinstance(policy_output, tuple):
                policy_output = policy_output[0]

            priors = policy_output.squeeze(0).detach().cpu().numpy()

        if was_training:
            self.policy_network.train()

        priors = np.asarray(priors, dtype=np.float32).reshape(-1)
        priors = priors[:self.action_space]
        priors = np.maximum(priors, 0.0)
        prior_sum = priors.sum()

        if len(priors) != self.action_space or not np.isfinite(prior_sum) or prior_sum <= 0:
            return np.ones(self.action_space, dtype=np.float32) / self.action_space

        return priors / prior_sum

    def predict_value(self):
        if self.value_network is None:
            return 0.0

        was_training = self.value_network.training
        self.value_network.eval()

        with torch.no_grad():
            state = self.encode_observation(self.observation, self.value_network)

            if hasattr(self.value_network, "get_value"):
                value = self.value_network.get_value(state)
            else:
                value = self.value_network(state)

        if was_training:
            self.value_network.train()

        if isinstance(value, torch.Tensor):
            value = value.squeeze().detach().cpu().item()

        return float(value)
    
    # Detach the parent node to save memory after the search is done
    def detach_parent(self):
        del self.parent
        self.parent = None

    # Clone the game environment state to be able to simulate future actions 
    def clone_env_state(self,game):
        # Use env_factory if available (for wrapped environments)
        if self.env_factory is not None:
            clone = self.env_factory()
            clone.reset()
        else:
            # Extract the base game name from the environment's spec
            base_game_name = game.unwrapped.spec.id
            clone = gym.make(base_game_name)
            clone.reset()
        
        src = game.unwrapped
        dst = clone.unwrapped

        if getattr(src, 'state', None) is not None:
            dst.state = np.array(src.state, dtype=np.float32).copy()

        if hasattr(src, 's'):
            dst.s = src.s

        if hasattr(src, 'lastaction'):
            dst.lastaction = src.lastaction

        if hasattr(src, 'steps_beyond_terminated'):
            dst.steps_beyond_terminated = src.steps_beyond_terminated
        
        if hasattr(game, '_elapsed_steps') and hasattr(clone, '_elapsed_steps'):
            clone._elapsed_steps = game._elapsed_steps
        
        return clone 

    def is_model_terminal(self, observation):
        """
        Infer termination for a predicted observation.

        Game-specific terminal logic should be passed by the runner through
        terminal_fn. If no terminal function is provided, model-based search
        treats predicted states as non-terminal.
        """

        if self.terminal_fn is not None:
            return self.terminal_fn(observation)

        return False

    def predict_model_step(self, observation, action):
        prediction = self.dynamics_model.predict(
            observation,
            action
        )

        if len(prediction) == 3:
            next_observation, reward, done = prediction
        else:
            next_observation, reward = prediction
            done = self.is_model_terminal(next_observation)

        return next_observation, reward, done
    
    # Create one child for each possible action of the game,
    # then apply such action to a copy of the current node environment 
    # and create such child node with proper information returned from the action executed.
    def create_child(self):

        if self.done:
            return
        
        child = {}
        policy_priors = self.get_policy_priors()

        for action in range(self.action_space):
            if self.dynamics_model is not None:
                game = self.clone_env_state(self.game)
                observation, reward, done = self.predict_model_step(
                    self.observation,
                    action
                )

                if getattr(game.unwrapped, 'state', None) is not None:
                    game.unwrapped.state = np.array(observation, dtype=np.float32).copy()

                if hasattr(game.unwrapped, 's'):
                    game.unwrapped.s = int(observation)

                if hasattr(game.unwrapped, 'lastaction'):
                    game.unwrapped.lastaction = action

                if hasattr(game, '_elapsed_steps'):
                    game._elapsed_steps += 1
            else:
                game = self.clone_env_state(self.game)
                step_out = game.step(action)

                if len(step_out) == 5:
                    observation, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    observation, reward, done, _ = step_out
            
            child[action] = MCTSNode(
                game,
                done,
                self,
                observation,
                action,
                self.game_name,
                self.c,
                env_factory=self.env_factory,
                dynamics_model=self.dynamics_model,
                policy_network=self.policy_network,
                value_network=self.value_network,
                observation_encoder=self.observation_encoder,
                prior_probability=float(policy_priors[action]),
                reward=reward,
                terminal_fn=self.terminal_fn,
                discount=self.discount
            )

        self.child = child

    # Rollout outputs a value for the current node.
    # If a value network is available, it evaluates the leaf directly.
    # Otherwise the old random environment rollout is used as a fallback.
    def rollout(self):

        v = self.reward

        if self.done:
            return v

        if self.value_network is not None:
            return v + self.discount * self.predict_value()
        
        done = False
        new_game = self.clone_env_state(self.game)
        discount = self.discount

        while not done:
            action = new_game.action_space.sample()
            step_out = new_game.step(action)

            if len(step_out) == 5:
                observation, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                observation, reward, done, _ = step_out

            v = v + discount * reward
            discount = discount * self.discount
            if done:
                new_game.close()
                break
        return v
    
    
    # From the current node, the children which maximize the value of the MCTS formula will be picked
    # At a leaf: if it was not explored before, a rollout will be done to get a value for the node
    #otherwise, expand the node by creating its children, pick one at random, do a rollout and update
    #backpropagate the value up to the root (meaning update value and visit counts)
    def explore(self):

        #find a leaf node by choosing nodes with max U

        current = self

        while current.child:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length", max_U)
            action = random.choice(actions)
            current = child[action]

        # play a random game, or expand if needed
        if current.N < 1:
            rollout_value = current.rollout()
            current.T = current.T + rollout_value
        else:
            current.create_child()
            if current.child:
                max_U = max(c.getUCBscore() for c in current.child.values())
                candidates = [
                    c for c in current.child.values()
                    if c.getUCBscore() == max_U
                ]
                current = random.choice(candidates)
            rollout_value = current.rollout()
            current.T = current.T + rollout_value

        current.N += 1

        #update statistics and backpropagate

        parent = current 

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T += rollout_value
    
    # after the search is done, the values should be statistically accurate.
    # this function will pick at random one of the node with highest visit count (should have a good value anyway)

    def next(self):

        if self.done:
            raise ValueError("game has ended")
        
        if not self.child:
            raise ValueError("no children found and game hasn\'t ended")
        
        child = self.child

        max_N = max(node.N for node in child.values())

        max_children = [c for a , c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index

    def visit_count_policy(self, temperature=1.0):
        if not self.child:
            raise ValueError("no children found and game hasn\'t ended")

        policy = np.zeros(self.action_space, dtype=np.float32)

        if temperature == 0:
            max_N = max(node.N for node in self.child.values())
            best_actions = [
                action for action, node in self.child.items()
                if node.N == max_N
            ]
            policy[random.choice(best_actions)] = 1.0
            return policy

        visits = np.array(
            [self.child[action].N for action in range(self.action_space)],
            dtype=np.float32
        )

        if temperature != 1.0:
            visits = visits ** (1.0 / temperature)

        visits_sum = visits.sum()
        if visits_sum <= 0:
            return np.ones(self.action_space, dtype=np.float32) / self.action_space

        policy[:] = visits / visits_sum
        return policy
    

#The agent can be used with different games
class MCTSAgent:

    """
    Initialize the Agent

    Args: 
    game_name: Name of the gym environment (for backward compatibility)
    env_factory: Callable that returns a fresh wrapped environment instance
    explore_iterations: Number of iterations per move 
    c: Exploration constant
    dynamics_model: Optional learned transition model used to expand nodes
    networks: Optional AlphaZeroNetworks container with policy/value networks
    policy_network: Optional policy network used for selection priors
    value_network: Optional value network used to evaluate rollout leaves
    observation_encoder: Optional callable that maps observations to NN inputs
    """
    def __init__(
        self,
        game_name=None,
        env_factory=None,
        explore_iterations=100,
        c=1.0,
        dynamics_model=None,
        networks=None,
        policy_network=None,
        value_network=None,
        observation_encoder=None,
        terminal_fn=None,
        discount=1.0
    ):

        self.game_name = game_name
        self.env_factory = env_factory
        self.explore_iterations = explore_iterations
        self.c = c
        self.current_tree = None
        self.dynamics_model = dynamics_model
        self.policy_network = (
            policy_network
            if policy_network is not None
            else getattr(networks, "policy_network", None)
        )
        self.value_network = (
            value_network
            if value_network is not None
            else getattr(networks, "value_network", None)
        )
        self.observation_encoder = observation_encoder
        self.terminal_fn = terminal_fn
        self.discount = discount

    """
    Initialize the Agent

    Args: 
    game: gym environment instance
    observation: current observation of the environment
    done: if the game is won/loss/draw

    Returns:
    action: the chosen action
    If return_policy is True, also returns the root visit-count policy target.
    """
    def get_action(self, game, observation, done, return_policy=False, temperature=1.0):
        
        #Initialize or update the tree
        if self.current_tree is None or done or self.dynamics_model is not None:
            new_game = self.clone_env(game)
            self.current_tree = MCTSNode(
                new_game,
                False,
                None,
                observation,
                0,
                self.game_name,
                self.c,
                env_factory=self.env_factory,
                dynamics_model=self.dynamics_model,
                policy_network=self.policy_network,
                value_network=self.value_network,
                observation_encoder=self.observation_encoder,
                terminal_fn=self.terminal_fn,
                discount=self.discount
            )

        # Perform MCTS search
        for i in range(self.explore_iterations):
            self.current_tree.explore()

        policy_target = None
        if return_policy:
            policy_target = self.current_tree.visit_count_policy(temperature)

        # Get best action
        next_tree, action = self.current_tree.next()
        next_tree.detach_parent()
        self.current_tree = next_tree

        if return_policy:
            return action, policy_target

        return action
    
    # Clone the environment state
    def clone_env(self, game):
        if self.env_factory is not None:
            # Use the factory to create a fresh wrapped environment
            clone = self.env_factory()
            clone.reset()
        else:
            # Original logic: vanilla game envs with no wrappers
            clone = gym.make(self.game_name)
            clone.reset()

        src = game.unwrapped
        dst = clone.unwrapped
        if getattr(src, 'state', None) is not None:
            dst.state = np.array(src.state, dtype=np.float32).copy()
        if hasattr(src, 's'):
            dst.s = src.s
        if hasattr(src, 'lastaction'):
            dst.lastaction = src.lastaction
        if hasattr(src, 'steps_beyond_terminated'):
            dst.steps_beyond_terminated = src.steps_beyond_terminated
        if hasattr(game, '_elapsed_steps') and hasattr(clone, '_elapsed_steps'):
            clone._elapsed_steps = game._elapsed_steps
        return clone
    
    # Reset the agent's internal tree
    def reset(self):
        self.current_tree = None
