import pytest
from unittest.mock import patch
import sprint1.MCTS_FrozenLake as MCTS_FrozenLake


@patch("sprint1.MCTS_FrozenLake.imageio.mimsave")
@patch("sprint1.MCTS_FrozenLake.MCTSAgent.get_action", return_value=0)

# Checks if main() runs without crashing
def test_frozenlake_main_runs_without_error(mock_get_action, mock_mimsave):
    
    try:
        MCTS_FrozenLake.main()
    except Exception as e:
        pytest.fail(f"main() raised an unexpected exception: {e}")


@patch("sprint1.MCTS_FrozenLake.imageio.mimsave")
@patch("sprint1.MCTS_FrozenLake.MCTSAgent.get_action", return_value=0)

# Checks if the agent is actually asked for actions
def test_frozenlake_main_calls_get_action(mock_get_action, mock_mimsave):
 
    MCTS_FrozenLake.main()

    assert mock_get_action.called


@patch("sprint1.MCTS_FrozenLake.imageio.mimsave")
@patch("sprint1.MCTS_FrozenLake.MCTSAgent.get_action", return_value=0)

# Checks if GIF saving is triggered
def test_frozenlake_main_saves_gif(mock_get_action, mock_mimsave):
   
    MCTS_FrozenLake.main()

    mock_mimsave.assert_called_once()



@patch("sprint1.MCTS_FrozenLake.imageio.mimsave")
@patch("sprint1.MCTS_FrozenLake.MCTSAgent.reset")
@patch("sprint1.MCTS_FrozenLake.MCTSAgent.get_action", return_value=0)

# Checks if the agent reset function is called
def test_frozenlake_main_resets_agent(
    mock_get_action,
    mock_reset,
    mock_mimsave
):
    
    MCTS_FrozenLake.main()

    assert mock_reset.called


