import unittest

import torch

from alphazero.value_and_policy_NN import AlphaZeroNetworks, ActionSpaceType


class AlphaZeroNetworksInputDimTest(unittest.TestCase):
    def test_accepts_game_specific_input_dim_for_discrete_actions(self):
        net = AlphaZeroNetworks(
            action_space_type=ActionSpaceType.DISCRETE,
            action_dim=5,
            hidden_states=32,
            input_dim=7,
            device='cpu',
        )

        state = torch.randn(2, 7)

        probs = net.policy_network(state)
        value = net.value_network(state)

        self.assertEqual(probs.shape, (2, 5))
        self.assertEqual(value.shape, (2, 1))

    def test_accepts_game_specific_input_dim_for_continuous_actions(self):
        net = AlphaZeroNetworks(
            action_space_type=ActionSpaceType.CONTINUOUS,
            action_dim=3,
            hidden_states=32,
            input_dim=11,
            device='cpu',
        )

        state = torch.randn(2, 11)

        mean, log_std = net.policy_network(state)
        value = net.value_network(state)

        self.assertEqual(mean.shape, (2, 3))
        self.assertEqual(log_std.shape, (2, 3))
        self.assertEqual(value.shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
