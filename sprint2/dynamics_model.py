import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """
    Generic feed-forward dynamics network.

    The network itself only needs:
        input_dim  = size of the full input vector
        output_dim = size of the full output vector

    For CartPole dynamics training we use:
        input  = [state, one_hot_action]
        output = [next_state, reward]

    state_dim and action_dim are optional metadata used by the convenience
    methods that split outputs and one-hot encode actions.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        state_dim=None,
        action_dim=None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward_raw(self, model_input):
        """
        Forward pass for an already concatenated input tensor.
        """

        x = F.relu(self.layer1(model_input))
        x = F.relu(self.layer2(x))

        return self.output_layer(x)

    def forward(self, state, action=None):
        """
        Forward pass.

        If action is provided, state and action are concatenated first and the
        output is split into (pred_next_state, pred_reward).

        If action is None, state is treated as the full model input and the raw
        output tensor is returned.
        """

        if action is None:
            return self.forward_raw(state)

        model_input = torch.cat([state, action], dim=1)
        output = self.forward_raw(model_input)

        if self.state_dim is None:
            return output

        pred_next_state = output[:, :self.state_dim]
        pred_reward = output[:, self.state_dim:]

        return pred_next_state, pred_reward

    def compute_loss(
        self,
        state,
        action,
        true_next_state,
        true_reward
    ):
        """
        MSE loss for next-state and reward prediction.
        """

        total_loss, _, _ = self.loss_details(
            state,
            action,
            true_next_state,
            true_reward
        )

        return total_loss

    def loss_details(
        self,
        state,
        action,
        true_next_state,
        true_reward
    ):
        """
        Return total loss plus separate state and reward losses for logging.
        """

        pred_next_state, pred_reward = self.forward(
            state,
            action
        )

        if len(true_reward.shape) == 1:
            true_reward = true_reward.unsqueeze(1)

        state_loss = F.mse_loss(
            pred_next_state,
            true_next_state
        )

        reward_loss = F.mse_loss(
            pred_reward,
            true_reward
        )

        total_loss = state_loss + reward_loss

        return total_loss, state_loss, reward_loss

    def one_hot_action(self, action):
        """
        Convert an integer action to a one-hot tensor.
        """

        if self.action_dim is None:
            raise ValueError("action_dim is needed to one-hot encode actions")

        if isinstance(action, torch.Tensor):
            action = int(action.item())

        vec = torch.zeros(self.action_dim)
        vec[action] = 1.0

        return vec

    @torch.no_grad()
    def predict(self, state, action, device=None):
        """
        Predict next state and reward from numpy state and integer action.
        """

        if self.state_dim is None:
            raise ValueError("state_dim is needed to split next_state and reward")

        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        action_tensor = self.one_hot_action(
            action
        ).unsqueeze(0).to(device)

        pred_next_state, pred_reward = self.forward(
            state_tensor,
            action_tensor
        )

        next_state = (
            pred_next_state
            .squeeze(0)
            .cpu()
            .numpy()
        )

        reward = (
            pred_reward
            .squeeze(0)
            .item()
        )

        if was_training:
            self.train()

        return next_state, reward
