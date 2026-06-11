import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional

"""Class for actionspace type, atm either discrete or continuous"""
class ActionSpaceType:
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'

"""Container class for the value and policy networks used in AlphaZero
uses discrete and continuous action spaces"""
class AlphaZeroNetworks:

    def __init__(self, 
                 action_space_type: str = ActionSpaceType.DISCRETE,
                 action_dim: Union[int, Tuple[int, ...]] = None,
                 hidden_states: int = 64,
                 continuous_action_bounds: Optional[Tuple[float, float]] = (-1, 1),
                 device: str = 'cuda'):
        
        """
        Initialize the NNs for a specific game.
        
        Args: 
            action_space_type: Type of action space (discrete or continuous).
            action_dim: Dimension of the action space (int for discrete, int or tuple for continuous).
            hidden_states: Number of hidden states in the networks.
            continuous_action_bounds: Bounds for continuous actions (min, max).
            device: Device to run the networks on ('cuda' or 'cpu').
        """

        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.hidden_states = hidden_states
        self.continuous_action_bounds = continuous_action_bounds
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

        if action_space_type == ActionSpaceType.DISCRETE:
            assert isinstace(action_dim, int), "For discrete action space, action_dim should be an integer."

            self.policy_network = DiscretePolicyNetwork(action_dim, hidden_states).to(self.device)
        elif action_space_type == ActionSpaceType.CONTINUOUS:
            assert action_dim is not None, "For continuous action space, action_dim should be specified."

            if isinstance(action_dim, int):
                action_dim = (action_dim,)
            self.policy_network = ContinuousPolicyNetwork(action_dim, hidden_states, continuous_action_bounds).to(self.device)

        else:
            raise ValueError("Invalid action space type. Must be 'discrete' or 'continuous'.")
        
        self.value_network = ValueNetwork(hidden_states).to(self.device)

    def to(self, device: str):
        """Move the networks to the specified device."""
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.policy_network = self.policy_network.to(device)
        self.value_network = self.value_network.to(device)
        return self
    
    def train(self):
        """Set the networks to training mode."""
        self.policy_network.train()
        self.value_network.train()

    def eval(self):
        """Set the networks to evaluation mode."""
        self.policy_network.eval()
        self.value_network.eval()

    def save(self, path_prefix):
        """Save the networks' state dictionaries."""
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'action_space_type': self.action_space_type,
            'action_dim': self.action_dim,
            'hidden_states': self.hidden_states,
            'continuous_action_bounds': self.continuous_action_bounds
        }, f"{path_prefix}_models.pth")

    def load(self, path_prefix):
        """Load the networks' state dictionaries."""
        checkpoint = torch.load(f"{path_prefix}_models.pth", map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])

    def get_action(self, state, deterministic: bool = False):
        """
        Get an action from the policy network given the actual state.
        
        Args:
            state: The current state.
            deterministic: Whether to select the most likely action (True) or sample from the distribution (False).
        
        Returns:
            The selected action.
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            state = state.to(self.device)

            return self.policy_network.get_action(state, deterministic)
        
class DiscretePolicyNetwork(nn.Module):
    """
    Policy Network for discrete action spaces. Takes the state as input and outputs a probability distribution over actions.
    """
    def __init__(self, num_actions: int, hidden_states: int = 64):
        super(DiscretePolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.hidden_states = hidden_states

        self.dense1 = nn.Linear(hidden_states, hidden_states)
        self.dense2 = nn.Linear(hidden_states, hidden_states)
        self.dense3 = nn.Linear(hidden_states, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode = 'fan_in', nonlinearity = 'relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.log_softmax(self.dense3(x), dim=-1)
        return torch.exp(x)  # Return probabilities
    
    def get_action(self, state, deterministic: bool = False):
        """Choose an action based on the policy."""
        probs = self.forward(state)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.argmax(probs, dim=-1)  # For now, we can use argmax for both cases. Sampling can be added later if needed.
        
        return action if deterministic else action
    
    def get_log_probs(self, state, actions):
        """Get log probabilities of the selected actions."""
        logits = self.dense2(F.relu(self.dense1(state)))
        logits = F.log_softmax(self.dense3(logits), dim=-1)
        return logits.gather(1, actions.unsqueeze(-1)).squeeze(-1)