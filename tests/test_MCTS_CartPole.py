import pytest
from unittest.mock import patch
import sprint1.MCTS_CartPole as MCTS_CartPole


# Run only 1 episode 
@patch("sprint1.MCTS_CartPole.num_episodes", 1)
@patch("sprint1.MCTS_CartPole.max_steps", 5)
@patch("sprint1.MCTS_CartPole.VideoFileClip")

# Skip MCTS computation
@patch("sprint1.MCTS_CartPole.MCTSAgent.get_action", return_value=0)



# Checks if main() runs without crashing
def test_cartpole_main_runs_without_error(
    mock_get_action,
    mock_videofileclip,
):
    try:
        MCTS_CartPole.main()
    except Exception as e:
        pytest.fail(f"main() raised an unexpected exception: {e}")


@patch("sprint1.MCTS_CartPole.num_episodes", 1)
@patch("sprint1.MCTS_CartPole.max_steps", 5)
@patch("sprint1.MCTS_CartPole.VideoFileClip")
@patch("sprint1.MCTS_CartPole.MCTSAgent.get_action", return_value=0)

# Checks if the agent is actually asked for actions
def test_cartpole_main_calls_get_action(
    mock_get_action,
    mock_videofileclip,
):

    MCTS_CartPole.main()

    assert mock_get_action.called


@patch("sprint1.MCTS_CartPole.num_episodes", 1)
@patch("sprint1.MCTS_CartPole.max_steps", 5)
@patch("sprint1.MCTS_CartPole.VideoFileClip")
@patch("sprint1.MCTS_CartPole.MCTSAgent.get_action", return_value=0)

# Checks if video export is triggered
def test_cartpole_main_creates_gif(
    mock_get_action,
    mock_videofileclip,
):

    MCTS_CartPole.main()

    mock_videofileclip.assert_called_once_with(
        "videos/rl-video-episode-0.mp4"
    )
    


@patch("sprint1.MCTS_CartPole.num_episodes", 1)
@patch("sprint1.MCTS_CartPole.max_steps", 5)
@patch("sprint1.MCTS_CartPole.VideoFileClip")
@patch("sprint1.MCTS_CartPole.MCTSAgent.reset")
@patch("sprint1.MCTS_CartPole.MCTSAgent.get_action", return_value=0)

# Checks if the agent reset function is called
def test_cartpole_main_resets_agent(
    mock_get_action,
    mock_reset,
    mock_videofileclip,
):

    MCTS_CartPole.main()

    assert mock_reset.called


