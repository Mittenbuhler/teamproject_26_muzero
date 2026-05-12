import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sprint1.mcts_agent import MCTSAgent

def test_agent_initialization():
    # checking if the constructor sets the given parameters correctly
    agent = MCTSAgent(game_name='CartPole-v1', explore_iterations=1, c=1.0) 
    assert agent.explore_iterations == 1  # check if it stored the search budget
    assert agent.c == 1.0 # check if exploration constant is set
    assert agent.current_tree is None # tree should be empty before the first search

@patch('sprint1.mcts_agent.gym.make')
def test_get_action_returns_valid_action(mock_gym_make):
    # check if the agent can return a valid action from the action space
    # need 2 iterations: 1st one does the rollout, 2nd one actually expands the children
    # otherwise next() fails because self.child is still None
    agent = MCTSAgent(game_name='CartPole-v1', explore_iterations=2)
    
    # mock environment setup to avoid side effects
    mock_env = MagicMock()
    mock_env.action_space.n = 2 # env has 2  actions (left/right)
    mock_env.observation_space.shape = (4,) # state vector size for cartpole (pos, vel, angle, etc.)
    mock_env.unwrapped = mock_env
    
    observation = np.array([0.1, 0.2, 0.3, 0.4]) 
    mock_gym_make.return_value = mock_env
    
    # mock step() to return done=True so the MCTS simulation terminates immediately
    # prevents the test from hanging in a long rollout loop  (very important!!!!)
    mock_env.step.return_value = (
        np.array([0.15, 0.25, 0.35, 0.45]), 1.0, True, False, {}
    )
    
    mock_env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
    
    # trigger the search process and get the recommended move
    action = agent.get_action(mock_env, observation, done=False)
    
    # final check: did we get a valid button index back?
    assert action in [0, 1] 

@patch('sprint1.mcts_agent.gym.make')
def test_reset_clears_the_tree(mock_gym_make):
    # testing if the reset method actually wipes the internal search tree
    agent = MCTSAgent(game_name='CartPole-v1', explore_iterations=2)
    
    # standard mock setup
    mock_env = MagicMock()
    mock_env.action_space.n = 2
    mock_env.observation_space.shape = (4,)
    mock_env.unwrapped = mock_env
    
    observation = np.array([0.1, 0.2, 0.3, 0.4])
    mock_gym_make.return_value = mock_env
    mock_env.step.return_value = (
        np.array([0.15, 0.25, 0.35, 0.45]), 1.0, True, False, {}
    )
    mock_env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
    
    # build a tree first by calling get_action so we have something to delete
    agent.get_action(mock_env, observation, done=False)
    assert agent.current_tree is not None # tree should exist in memory now
    
    # clear it and verify the reference is gone 
    agent.reset()
    assert agent.current_tree is None # should be null