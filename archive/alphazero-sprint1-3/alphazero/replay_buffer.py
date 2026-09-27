from collections import deque, namedtuple
import random


'''
The ReplayBuffer stores game plays that we will use for neural network training. It stores, in particular:
    - The observation (i.e. state) of the game environment
    - The target Value
    - The observation (i.e. state) of the game environment at the previous step
    - The target Policy according to visit counts 
'''

class ReplayBuffer:

    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
    
        """Initialize a ReplayBuffer object.
        Params:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observation", "value", "prev_obs", "policy"])
    
    def add(self, observation, value, prev_obs, policy):
    
        """Add a new experience to memory."""
        
        e = self.experience(observation, value, prev_obs, policy)
        self.memory.append(e)
    
    def sample(self):
    
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

    def __len__(self):
    
        """Return the current size of internal memory."""
        
        return len(self.memory)
    

