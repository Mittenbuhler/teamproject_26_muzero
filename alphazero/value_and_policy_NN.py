from functorch import dim
import torch
from torch.distributed import log
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
            assert isinstance(action_dim, int), "For discrete action space, action_dim should be an integer."

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

            if hasattr(self.policy_network, 'get_action'):
                return self.policy_network.get_action(state, deterministic)

            action, _ = self.policy_network.sample_action(state, deterministic)
            return action
        
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
                    nn.init.constant_(module.bias, 0)

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
    
class ContinuousPolicyNetwork(nn.Module):
    """
    Policy Network for continuous action spaces. Takes the state as input and outputs the mean and log std of a Gaussian distribution over actions.
    """
    def __init__(self, action_dim: Tuple[int, ...], hidden_states: int = 64, action_bounds: Tuple[float, float] = (-1, 1)):
        super(ContinuousPolicyNetwork, self).__init__()

        self.action_dim = action_dim
        self.action_size = int(torch.prod(torch.tensor(action_dim)))
        self.hidden_states = hidden_states
        self.action_bounds = action_bounds

        self.dense1 = nn.Linear(hidden_states, hidden_states)
        self.dense2 = nn.Linear(hidden_states, hidden_states)

        self.mean_head = nn.Linear(hidden_states, self.action_size)
        self.log_std_head = nn.Linear(hidden_states, self.action_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        nn.init.constant_(self.log_std_head.bias, -0.5)  # Initialize log std to a small value (std ~ 0.6)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input state tensor of shape (batch_size, hidden_states).

        Returns:
            mean: Mean of the Gaussian distribution over actions.
            log_std: Log standard deviation of the Gaussian distribution over actions.
        """
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp log std to prevent numerical issues

        mean = torch.tanh(mean) 
        min_bound, max_bound = self.action_bounds
        mean = min_bound + (mean + 1) * 0.5 * (max_bound - min_bound)  # Scale mean to action bounds
        
        return mean, log_std
    
    def get_action(self, state, deterministic: bool = False):
        """Compatibility wrapper for the shared action-selection interface."""
        action, _ = self.sample_action(state, deterministic)
        return action

    def sample_action(self, state, deterministic: bool = False):
        """Sample an action from the policy given the state.
        
        Args:
            state: The current state tensor of shape (batch_size, hidden_states).
            deterministic: Whether to select the mean action (True) or sample from the distribution (False).

        Returns:
            action: The sampled action tensor.
            log_prob: The log probability of the sampled action.
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # Reparameterization trick for backpropagation
            log_prob = normal.log_prob(action).sum(dim=-1)

        # Clamp action to bounds
        action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])

        if len(self.action_dim) > 1:
            action = action.view(-1, *self.action_dim)  # Reshape to original action dimensions

        return action, log_prob  
    
    def get_log_probs(self, state, actions):
        """Get log probabilities of the selected actions."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        return log_probs
    
    def entropy(self, state):
        """Calculate the entropy of the action distribution given the state."""
        _, log_std = self.forward(state)
        std = torch.exp(log_std)

        return torch.sum(torch.log(std * torch.sqrt(2 * torch.pi * torch.exp(1))), dim=-1)

class ValueNetwork(nn.Module):
    """
    Value Network. Takes the state as input and outputs a scalar value representing the expected return from that state.
    """
    def __init__(self, hidden_states: int = 64):
        super(ValueNetwork, self).__init__()

        self.hidden_states = hidden_states

        self.dense1 = nn.Linear(hidden_states, hidden_states)
        self.dense2 = nn.Linear(hidden_states, hidden_states)
        self.dense3 = nn.Linear(hidden_states, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        value = self.dense3(x)
        return torch.tanh(value)  # Output value in range [-1, 1]
    
    def get_value(self, state):
        """Get the value of the given state."""
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            state = state.to(next(self.parameters()).device)
            value = self.forward(state)

            return value.item() if value.shape[0] == 1 else value.squeeze(-1)
        
class CombinedNetwork(nn.Module):
    """
    Combined Network for both value and policy. This can be used to share parameters between the two networks if desired.
    """
    def __init__(self, action_space_type: str, action_dim: Union[int, Tuple[int, ...]] = None, hidden_states: int = 64, continuous_action_bounds: Optional[Tuple[float, float]] = (-1, 1)):
        super(CombinedNetwork, self).__init__()

        self.action_space_type = action_space_type
        self.action_dim = action_dim
        self.hidden_states = hidden_states

        self.shared_dense1 = nn.Linear(hidden_states, hidden_states)
        self.shared_dense2 = nn.Linear(hidden_states, hidden_states)

        self.value_head = nn.Linear(hidden_states, 1)

        if action_space_type == ActionSpaceType.DISCRETE:
            assert isinstance(action_dim, int), "For discrete action space, action_dim should be an integer."
            self.policy_head = nn.Linear(hidden_states, action_dim)
            self.policy_head_activation = nn.LogSoftmax(dim=-1)
        else:
            if isinstance(action_dim, int):
                action_dim = (action_dim,)
            self.action_size = int(torch.prod(torch.tensor(action_dim)))
            self.mean_head = nn.Linear(hidden_states, self.action_size)
            self.log_std_head = nn.Linear(hidden_states, self.action_size)
            self.continuous_action_bounds = continuous_action_bounds

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.shared_dense1(x))
        x = F.relu(self.shared_dense2(x))

        value = torch.tanh(self.value_head(x))  # Value output in range [-1, 1]

        if self.action_space_type == ActionSpaceType.DISCRETE:
            policy = self.policy_activation(self.policy_head(x))  # Log probabilities for discrete actions
            return policy, value
        else:
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, -20, 2)  # Clamp log std to prevent numerical issues

            mean = torch.tanh(mean) 
            min_bound, max_bound = self.continuous_action_bounds
            mean = min_bound + (mean + 1) * 0.5 * (max_bound - min_bound)  # Scale mean to action bounds

        return (mean, log_std), value
    



# Beispiel für die Verwendung
if __name__ == "__main__":
    HIDDEN_STATES = 64
    
    # Beispiel 1: Diskreter Aktionsraum (z.B. Tic-Tac-Toe, Schach, Go)
    print("=== Diskreter Aktionsraum ===")
    discrete_net = AlphaZeroNetworks(
        action_space_type='discrete',
        action_dim=9,  # 9 mögliche Aktionen
        hidden_states=HIDDEN_STATES
    )
    
    sample_state = torch.randn(1, HIDDEN_STATES)
    action_probs = discrete_net.policy_network(sample_state)
    action = discrete_net.get_action(sample_state, deterministic=False)
    
    print(f"Aktionswahrscheinlichkeiten Shape: {action_probs.shape}")
    print(f"Gewählte Aktion: {action}")
    print(f"Zustandswert: {discrete_net.value_network(sample_state).item():.3f}")
    
    # Beispiel 2: Kontinuierlicher Aktionsraum (z.B. Robot Control, Pendulum)
    print("\n=== Kontinuierlicher Aktionsraum ===")
    continuous_net = AlphaZeroNetworks(
        action_space_type='continuous',
        action_dim=2,  # 2-dimensionale Aktion (z.B. Kraft in x und y)
        hidden_states=HIDDEN_STATES,
        continuous_action_bounds=(-2.0, 2.0)
    )
    
    continuous_action = continuous_net.get_action(sample_state, deterministic=False)
    print(f"Kontinuierliche Aktion: {continuous_action}")
    print(f"Aktions-Shape: {continuous_action.shape}")
    
    # Entropie für Exploration (kontinuierlich)
    mean, log_std = continuous_net.policy_network(sample_state)
    print(f"Mittelwert: {mean.squeeze().detach().numpy()}")
    print(f"Std: {torch.exp(log_std).squeeze().detach().numpy()}")