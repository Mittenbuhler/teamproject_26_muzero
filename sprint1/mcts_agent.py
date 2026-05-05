import numpy as np
import gym
import random
from math import sqrt, log

# MCTS Node class represents a node of the tree and contains information needed for
# the algorithm to run its search.
class MCTSNode:
    def __init__(self, game, done, parent, observation, action_index, game_name, c=1.0):

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

        # game name
        self.game_name = game_name

        # exploration constant
        self.c = c

        # get the info from the game environment
        if game is not None:
            self.action_space = game.action_space.n
            self.game_obs = game.observation_space.shape[0]

    # getUCBscore is the formula that gives a value to the node.
    # MCTS will pick the nodes with the highest value.        
    def getUCBscore(self):

        # Unexplored nodes get a max value to favour exploration
        if self.N == 0:
            return float('inf')
        
        # Get information about the parent node of current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        
        # Use one of the possible MCTS formula for calculating the node value 
        return (self.T / self.N) + self.c * sqrt(log(top_node.N) / self.N)
    
    # Detach the parent node to save memory after the search is done
    def detach_parent(self):
        del self.parent
        self.parent = None

    # Clone the game environment state to be able to simulate future actions 
    def clone_env_state(self,game):
        clone = gym.make(self.game_name)
        clone.reset()
        src = game.unwrapped
        dst = clone.unwrapped

        if getattr(src, 'state', None) is not None:
            dst.state = np.array(src.state, dtype=np.float32).copy()

        if hasattr(src, 'steps_beyond_terminated'):
            dst.steps_beyond_terminated = src.steps_beyond_terminated
        
        if hasattr(game, '_elapsed_steps') and hasattr(clone, '_elapsed_steps'):
            clone._elapsed_steps = game._elapsed_steps
        
        return clone 
    
    # Create one child for each possible action of the game,
    # then apply such action to a copy of the current node environment 
    # and create such child node with proper information returned from the action executed.
    def create_child(self):

        if self.done:
            return
        
        child = {}

        for action in range(self.game_actions):
            game = self.clone_env_state(self.game)
            step_out = game.step(action)

            if len(step_out) == 5:
                observation, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                observation, reward, done, _ = step_out
            
            child[action] = MCTSNode(game, done, self, observation, action, self.game_name, self.c)

        self.child = child

    # Rollout is a random play from a copy of the environment of the current node.
    # It will output a value for the current node.
    # -> The value is first random, but the more rollouts the more accurate is the average of the value
    # for the node. (Core of MCTS algorithm)
    def rollout(self):

        if self.done:
            return 0
        
        v = 0
        done = False
        new_game = self.clone_env_state(self.game)

        while not done:
            action = new_game.action_space.sample()
            step_out = new_game.step(action)

            if len(step_out) == 5:
                observation, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                observation, reward, done, _ = step_out

            v = v + reward
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
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.T = current.T + current.rollout()

        current.N += 1

        #update statistics and backpropagate

        parent = current 

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T += current.T