from replay_buffer import ReplayBuffer
from value_and_policy_NN import AlphaZeroNetworks, ActionSpaceType
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from policy_player_MCTS import Policy_Player_MCTS
from mcts_agent_policyValue import MCTSNode

BUFFER_SIZE = 100   
BATCH_SIZE = 32         
UPDATE_EVERY = 1
LOSS_AVG_WINDOW = 10



def train(config):
    rewards = []
    moving_average = []

    v_losses = []
    p_losses = []


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the replay buffer
    replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    # Create the AlphaZero networks with discrete action space
    networks = AlphaZeroNetworks(
        action_space_type=ActionSpaceType.DISCRETE,
        action_dim=config["action_dim"],  
        hidden_states=64,
        input_dim=config["input_dim"],   
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
    policy_loss_epsilon = 1e-8

   
    for e in range(config["episodes"]):

        reward_e = 0    
        game = config["make_env"]()
        reset_out = game.reset()
        
        # Handle both old gym and new gymnasium API
        if isinstance(reset_out, tuple):
            observation, _ = reset_out
        else:
            observation = reset_out
        
        done = False
        
        new_game = deepcopy(game)
        mytree = MCTSNode(
            new_game,
            False,
            0,
            observation,
            0,
            config["game_name"],
            env_factory=config["make_env"],
            policy_network=policy_network,
            value_network=value_network
        )
        
        print(f'Episode {e+1}/{config["episodes"]}')
        
        observations = []
        policies = []
        prev_observations = []
        step_count = 0
        
        
        while not done:
            
            if mytree.done:
                print(f"  Warning: MCTS tree marked as done prematurely. Resetting...")
                new_game = deepcopy(game)
                mytree = MCTSNode(
                    new_game,
                    False,
                    0,
                    observation,
                    0,
                    config["game_name"],
                    env_factory=config["make_env"],
                    policy_network=policy_network,
                    value_network=value_network
                )
        
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
            step_count += 1
        
                    
            if done:
                for i in range(len(observations)):
                    replay_buffer.add(observations[i], reward_e, prev_observations[i], policies[i])
                game.close()
                break
            
        print(f'Steps: {step_count},  Final reward: {reward_e}')
        rewards.append(reward_e)
        moving_average.append(np.mean(rewards[-100:]))
        
        if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:   
            
            
            # update and train the neural networks     
            experiences = replay_buffer.sample()
                
            # Each state has as target value the total rewards of the episode
            inputs = np.array([experience.observation for experience in experiences])
            targets = np.array([[experience.value / config["max_reward"]] for experience in experiences])
            
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
            targets_tensor = torch.FloatTensor(targets).to(device)
            
            # Train policy network
            policy_network.train()
            optimizer_p.zero_grad()
            policy_preds = policy_network(inputs_tensor)
            loss_p = -(
                targets_tensor * torch.log(policy_preds + policy_loss_epsilon)
            ).sum(dim=1).mean()
            loss_p.backward()
            optimizer_p.step()
                            
            p_losses.append(loss_p.item())

            avg_v_loss = np.mean(v_losses[-LOSS_AVG_WINDOW:])
            avg_p_loss = np.mean(p_losses[-LOSS_AVG_WINDOW:])
            print(
                "NN update: "
                f"value_loss={loss_v.item():.4f} "
                f"(avg{LOSS_AVG_WINDOW}={avg_v_loss:.4f}), "
                f"policy_loss={loss_p.item():.4f} "
                f"(avg{LOSS_AVG_WINDOW}={avg_p_loss:.4f}), "
                f"reward_avg100={moving_average[-1]:.2f}, "
                f"buffer={len(replay_buffer)}/{BUFFER_SIZE}"
            )
        else:
            print(
                "NN update: waiting for replay buffer "
                f"({len(replay_buffer)}/{BATCH_SIZE + 1} samples needed)"
            )

    return rewards, moving_average, v_losses, p_losses
