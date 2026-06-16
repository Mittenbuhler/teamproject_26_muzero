from IPython.display import clear_output
from replay_buffer import ReplayBuffer
from value_and_policy_NN import AlphaZeroNetworks, ActionSpaceType
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from policy_player_MCTS import Policy_Player_MCTS
from mcts_agent_policyValue import MCTSNode

BUFFER_SIZE = int(100)   # replay buffer size
BATCH_SIZE = 32         # minibatch size
UPDATE_EVERY = 1

episodes = 10

rewards = []
moving_average = []
v_losses = []
p_losses = []

# the maximum reward of the current game to scale the values
MAX_REWARD = 500

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

# Create the AlphaZero networks with discrete action space
# Assuming CartPole-v1 with 2 actions
networks = AlphaZeroNetworks(
    action_space_type=ActionSpaceType.DISCRETE,
    action_dim=2,  # CartPole has 2 actions
    hidden_states=64,
    input_dim=4,   # CartPole has 4 state dimensions
    device=device.type
)

# Extract the networks
value_network = networks.value_network   # estimates the total_reward from a given state
policy_network = networks.policy_network # estimates the probability of actions from a given state

# Create optimizers (updates the weights of the networks based on the loss)
optimizer_v = optim.Adam(value_network.parameters(), lr=0.001)
optimizer_p = optim.Adam(policy_network.parameters(), lr=0.001)

# Loss functions
mse_loss = nn.MSELoss()
crossentropy_loss = nn.CrossEntropyLoss()

# Game configuration
GAME_NAME = 'CartPole-v1'  # Or change to other gym environments as needed

'''
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our AlphaZero algorithm
- we will collect and plot the rewards to check if the AlphaZero is actually working.
- For CartPole-v1, in particular, 500 is the maximum possible reward. 
'''

for e in range(episodes):

    reward_e = 0    
    game = gym.make(GAME_NAME)
    reset_out = game.reset()
    
    # Handle both old gym and new gymnasium API
    if isinstance(reset_out, tuple):
        observation, _ = reset_out
    else:
        observation = reset_out
    
    done = False
    
    new_game = deepcopy(game)
    mytree = MCTSNode(new_game, False, 0, observation, 0, GAME_NAME)
    
    print(f'Episode {e+1}/{episodes}', flush=True)
    
    observations = []
    policies = []
    prev_observations = []
    
    step = 0
    
    while not done:
        
        step = step + 1
    
        mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)
        
        observations.append(ob)
        policies.append(p)
        prev_observations.append(p_ob)      
            
        step_out = game.step(action)
        
        # Handle both old gym (4 values) and new gymnasium (5 values) API
        if len(step_out) == 5:
            _, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            _, reward, done, _ = step_out
            
        reward_e = reward_e + reward
        
        #game.render()
                
        if done:
            for i in range(len(observations)):
                replay_buffer.add(observations[i], reward_e, prev_observations[i], policies[i])
            game.close()
            break
        
    print(f'  Final reward: {reward_e}', flush=True)
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
    
    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:   
        
        # clear output
        
        for i in range(10):
            clear_output(wait=True) 
        
        # update and train the neural networks
                
        experiences = replay_buffer.sample()
            
        # Each state has as target value the total rewards of the episode
            
        inputs = np.array([experience.observation for experience in experiences])
        targets = np.array([[experience.value / MAX_REWARD] for experience in experiences])
        
        # Convert to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        targets_tensor = torch.FloatTensor(targets).to(device)
        
        # Train value network
        value_network.train()
        optimizer_v.zero_grad()
        value_preds = value_network(inputs_tensor)
        loss_v = mse_loss(value_preds, targets_tensor)
        loss_v.backward()
        optimizer_v.step()
                        
        v_losses.append(loss_v.item())
        
        # Each state has as target policy the policy according to visit counts
            
        inputs = np.array([experience.prev_obs for experience in experiences])
        targets = np.array([experience.policy for experience in experiences])
        
        # Convert to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        # For policy: targets need to be action indices for CrossEntropyLoss
        # Assuming targets are already in the correct format
        targets_tensor = torch.FloatTensor(targets).to(device)
        
        # Train policy network
        policy_network.train()
        optimizer_p.zero_grad()
        policy_preds = policy_network(inputs_tensor)
        # Use CrossEntropyLoss with softmax output
        loss_p = crossentropy_loss(policy_preds, targets_tensor)
        loss_p.backward()
        optimizer_p.step()
                        
        p_losses.append(loss_p.item())