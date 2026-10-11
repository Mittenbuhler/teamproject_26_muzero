"""A deliberately small, configurable actor/learner implementation of MuZero."""
import argparse
import math
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "cartpole_muzero_matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .buffers import LatentReplayBuffer
from .environment import make_state_env
from .mcts import ModelBasedMCTS, select_action, visit_count_policy
from .models import DynamicsModel, RepresentationNetwork, PredictionNetwork

CHECKPOINT_VERSION = 15
OBSERVATION_MODE = "cartpole_state_4d"
POLICY_TARGET_MODE = "raw_visit_counts"
WARMUP_POLICY_MODE = "masked_policy_with_active_model_learning"


def save_reward_plot(history, path, moving_average_window=10):
    """Save raw training scores and deterministic MCTS evaluation scores."""
    rewards = np.asarray(history.get("scores", []), dtype=np.float32)
    if rewards.size == 0:
        raise ValueError("Cannot plot an empty reward history")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, rewards.size + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        episodes,
        rewards,
        color="#8bb8e8",
        alpha=0.55,
        label="training reward (raw)",
    )
    window = min(int(moving_average_window), rewards.size)
    if window > 1:
        average = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axis.plot(
            episodes[window - 1 :],
            average,
            color="#145da0",
            linewidth=2,
            label=f"{window}-episode average",
        )
    evaluations = history.get("evaluations", [])
    if evaluations:
        evaluation_episodes, evaluation_rewards = zip(*evaluations)
        axis.plot(
            evaluation_episodes,
            evaluation_rewards,
            "o-",
            color="#d1495b",
            label="MCTS evaluation reward (raw)",
        )
    axis.set(title="MuZero environment reward", xlabel="Episode", ylabel="Raw reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def scale_gradient(tensor, scale):
    """Scale only the backward gradient while preserving the forward value."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


def temperature_for_episode(
    episode,
    initial=1.0,
    middle=0.5,
    final=0.25,
    initial_episodes=1,
    middle_episodes=1,
):
    """Piecewise-constant action-selection temperature."""
    if episode < initial_episodes:
        return float(initial)
    if episode < middle_episodes:
        return float(middle)
    return float(final)


def select_training_action(root, temperature):
    """Sample behavior with temperature but train on raw root visit counts."""
    action, _ = select_action(root, temperature)
    target_policy = visit_count_policy(root, temperature=1.0)
    return action, target_policy


def updates_for_episode(
    trajectory_length,
    updates_per_episode,
    updates_per_transition,
    min_updates_per_episode,
    max_updates_per_episode,
):
    """Resolve fixed or transition-relative learner update frequency."""
    if updates_per_transition <= 0:
        return int(updates_per_episode)
    updates = math.ceil(trajectory_length * updates_per_transition)
    return int(np.clip(updates, min_updates_per_episode, max_updates_per_episode))


def bootstrapped_value_targets(
    trajectory,
    discount=0.997,
    bootstrap_steps=10,
    reward_scale=1.0,
):
    """Compute scalar n-step targets in scaled-reward units."""
    targets = []
    for start in range(len(trajectory)):
        value = 0.0
        discount_power = 1.0
        terminated = False
        for offset in range(bootstrap_steps):
            index = start + offset
            if index >= len(trajectory):
                terminated = True
                break
            value += (
                discount_power
                * float(trajectory[index]["reward"])
                * reward_scale
            )
            discount_power *= discount
            if trajectory[index].get("terminated", False) or trajectory[index].get(
                "truncated", False
            ):
                terminated = True
                break
        bootstrap_index = start + bootstrap_steps
        if not terminated and bootstrap_index < len(trajectory):
            # Search values already use the network's scaled reward/value units.
            value += discount_power * float(trajectory[bootstrap_index]["value"])
        targets.append(float(value))
    return targets


def train_latent_step(
    representation,
    dynamics,
    prediction,
    buffer,
    optimizer,
    batch_size,
    unroll_steps,
    action_dim,
    device,
    gradient_clip_norm=5.0,
    dynamics_gradient_scale=0.5,
    policy_loss_weight=1.0,
    value_loss_weight=1.0,
    reward_loss_weight=1.0,
):
    """Train one root plus exactly ``unroll_steps`` recurrent predictions."""
    (
        observations,
        actions,
        target_policies,
        target_values,
        target_rewards,
        policy_masks,
        value_masks,
        reward_masks,
    ) = buffer.sample(batch_size, unroll_steps, action_dim, device)

    optimizer.zero_grad()
    state = representation(observations.float())
    policy_sum = torch.zeros((), device=device)
    value_sum = torch.zeros((), device=device)
    reward_sum = torch.zeros((), device=device)
    prediction_calls = 0

    for depth in range(unroll_steps + 1):
        logits, value = prediction(state)
        prediction_calls += 1
        policy_per_sample = -(
            target_policies[:, depth] * F.log_softmax(logits, dim=-1)
        ).sum(dim=-1, keepdim=True)
        policy_sum += (policy_per_sample * policy_masks[:, depth]).sum()
        value_sum += (
            F.mse_loss(value, target_values[:, depth], reduction="none")
            * value_masks[:, depth]
        ).sum()
        if depth < unroll_steps:
            state, reward = dynamics(
                scale_gradient(state, dynamics_gradient_scale),
                actions[:, depth],
            )
            reward_sum += (
                F.mse_loss(reward, target_rewards[:, depth], reduction="none")
                * reward_masks[:, depth]
            ).sum()

    policy_loss = policy_sum / policy_masks.sum().clamp_min(1)
    value_loss = value_sum / value_masks.sum().clamp_min(1)
    reward_loss = reward_sum / reward_masks.sum().clamp_min(1)
    total_loss = (
        policy_loss_weight * policy_loss
        + value_loss_weight * value_loss
        + reward_loss_weight * reward_loss
    )
    total_loss.backward()
    parameters = [
        *representation.parameters(),
        *dynamics.parameters(),
        *prediction.parameters(),
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        gradient_clip_norm,
    )
    optimizer.step()
    return {
        "total_loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "gradient_norm": float(gradient_norm),
        "prediction_calls": prediction_calls,
    }


def make_latent_mcts(
    dynamics,
    prediction,
    action_dim,
    simulations=25,
    discount=0.997,
    pb_c_base=19652,
    pb_c_init=1.25,
    root_dirichlet_alpha=0.25,
    root_exploration_fraction=0.25,
):
    return ModelBasedMCTS(
        dynamics,
        prediction,
        action_dim,
        simulations=simulations,
        discount=discount,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )


def companion_checkpoint_path(path, kind):
    path = Path(path)
    return path.with_name(path.stem + f"_{kind}" + path.suffix)


def save_latent_checkpoint(path, representation, dynamics, prediction, metadata=None):
    """Atomically write a complete checkpoint in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "representation_state_dict": representation.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "prediction_state_dict": prediction.state_dict(),
        "observation_mode": OBSERVATION_MODE,
        "input_dim": representation.input_dim,
        "latent_dim": representation.latent_dim,
        "representation_hidden_dim": representation.hidden_dim,
        "dynamics_hidden_dim": dynamics.hidden_dim,
        "prediction_hidden_dim": prediction.hidden_dim,
        "action_dim": prediction.action_dim,
        "policy_target_mode": POLICY_TARGET_MODE,
        "warmup_policy_mode": WARMUP_POLICY_MODE,
    }
    payload.update(metadata or {})
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_latent_checkpoint(path, device=None):
    """Load only the vector-input architecture; older image/state weights cannot resume."""
    device = device or torch.device("cpu")
    data = torch.load(path, map_location=device, weights_only=False)
    version = data.get("checkpoint_version", data.get("training_version"))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"incompatible legacy MuZero checkpoint version {version}; expected "
            f"{CHECKPOINT_VERSION} (4D input). Older screenshot and pre-v8 models "
            "have different architectures; start a fresh run."
        )
    if data.get("observation_mode") != OBSERVATION_MODE or data.get("input_dim") != 4:
        raise ValueError("checkpoint requires the cartpole_state_4d observation mode")
    if data.get("policy_target_mode") != POLICY_TARGET_MODE:
        raise ValueError("checkpoint policy-target metadata is missing or incompatible")
    if data.get("warmup_policy_mode") != WARMUP_POLICY_MODE:
        raise ValueError("checkpoint warm-up policy metadata is missing or incompatible")
    representation = RepresentationNetwork(
        data["latent_dim"], data["representation_hidden_dim"],
    ).to(device)
    dynamics = DynamicsModel(
        data["latent_dim"], data["action_dim"], data["dynamics_hidden_dim"],
    ).to(device)
    prediction = PredictionNetwork(
        data["latent_dim"], data["action_dim"], data["prediction_hidden_dim"],
    ).to(device)
    representation.load_state_dict(data["representation_state_dict"])
    dynamics.load_state_dict(data["dynamics_state_dict"])
    prediction.load_state_dict(data["prediction_state_dict"])
    return representation, dynamics, prediction, data


def evaluate(
    env_id,
    representation,
    prediction,
    mcts,
    episodes,
    max_steps,
    seeds,
    use_mcts=True,
    random_actions=False,
):
    """Evaluate on identical seeds without exploration noise or actor RNG side effects."""
    seeds = list(seeds)
    if episodes <= 0 or max_steps <= 0 or len(seeds) < episodes:
        raise ValueError("evaluation needs positive episodes/max_steps and one seed per episode")
    env = make_state_env(env_id)
    numpy_random_state = np.random.get_state()
    scores = []
    try:
        for seed in seeds[:episodes]:
            np.random.seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            score = 0.0
            for _ in range(max_steps):
                if random_actions:
                    action = env.action_space.sample()
                else:
                    latent = representation.encode(observation)
                    if use_mcts:
                        root = mcts.search(latent, add_exploration_noise=False)
                        action = select_action(root, temperature=0.0)[0]
                    else:
                        action = int(np.argmax(prediction.predict(latent)[0]))
                observation, reward, terminated, truncated, _ = env.step(action)
                score += reward
                if terminated or truncated:
                    break
            scores.append(score)
    finally:
        np.random.set_state(numpy_random_state)
        env.close()
    return float(np.mean(scores)), scores


def validate_training_arguments(**arguments):
    """Validate public training/CLI values before constructing an environment."""
    positive_integer_names = (
        "episodes",
        "max_steps",
        "latent_dim",
        "hidden_dim",
        "batch_size",
        "buffer_capacity",
        "bootstrap_steps",
        "unroll_steps",
        "simulations",
        "evaluation_interval",
        "evaluation_episodes",
        "max_updates_per_episode",
    )
    for name in positive_integer_names:
        if int(arguments[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(arguments["latent_dim"]) < 2:
        raise ValueError("latent_dim must be at least 2 for min-max normalization")
    if int(arguments["warmup_episodes"]) < 0:
        raise ValueError("warmup_episodes must be nonnegative")
    if int(arguments["updates_per_episode"]) < 0:
        raise ValueError("updates_per_episode must be nonnegative")
    if int(arguments["min_updates_per_episode"]) < 0:
        raise ValueError("min_updates_per_episode must be nonnegative")
    if arguments["min_updates_per_episode"] > arguments["max_updates_per_episode"]:
        raise ValueError("min_updates_per_episode cannot exceed max_updates_per_episode")
    if arguments["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if arguments["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    if arguments["gradient_clip_norm"] <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if not 0 <= arguments["dynamics_gradient_scale"] <= 1:
        raise ValueError("dynamics_gradient_scale must be in [0, 1]")
    if not 0 <= arguments["discount"] <= 1:
        raise ValueError("discount must be in [0, 1]")
    if arguments["reward_scale"] <= 0:
        raise ValueError("reward_scale must be positive")
    loss_weights = [
        arguments["policy_loss_weight"],
        arguments["value_loss_weight"],
        arguments["reward_loss_weight"],
    ]
    if any(weight < 0 for weight in loss_weights):
        raise ValueError("loss weights must be nonnegative")
    if not any(weight > 0 for weight in loss_weights):
        raise ValueError("at least one loss weight must be positive")
    if arguments["pb_c_base"] <= 0 or arguments["pb_c_init"] <= 0:
        raise ValueError("PUCT parameters must be positive")
    if arguments["root_dirichlet_alpha"] <= 0:
        raise ValueError("root_dirichlet_alpha must be positive")
    if not 0 <= arguments["root_exploration_fraction"] <= 1:
        raise ValueError("root_exploration_fraction must be in [0, 1]")
    temperatures = (
        arguments["temperature_initial"],
        arguments["temperature_middle"],
        arguments["temperature_final"],
    )
    if any(temperature < 0 for temperature in temperatures):
        raise ValueError("temperatures must be nonnegative")
    initial_episodes = arguments["temperature_initial_episodes"]
    middle_episodes = arguments["temperature_middle_episodes"]
    if not 0 <= initial_episodes <= middle_episodes:
        raise ValueError(
            "temperature episodes must satisfy 0 <= initial_episodes <= middle_episodes"
        )


def train_latent_muzero(
    env_id="CartPole-v1",
    episodes=10,
    max_steps=500,
    latent_dim=64,
    hidden_dim=128,
    simulations=25,
    batch_size=32,
    buffer_capacity=20000,
    warmup_episodes=1,
    updates_per_episode=5,
    updates_per_transition=0.0,
    min_updates_per_episode=1,
    max_updates_per_episode=50,
    learning_rate=3e-4,
    weight_decay=1e-5,
    gradient_clip_norm=5.0,
    dynamics_gradient_scale=0.5,
    discount=0.997,
    bootstrap_steps=10,
    unroll_steps=5,
    reward_scale=1.0,
    policy_loss_weight=1.0,
    value_loss_weight=1.0,
    reward_loss_weight=1.0,
    pb_c_base=19652,
    pb_c_init=1.25,
    root_dirichlet_alpha=0.25,
    root_exploration_fraction=0.25,
    temperature_initial=1.0,
    temperature_middle=0.5,
    temperature_final=0.25,
    temperature_initial_episodes=200,
    temperature_middle_episodes=500,
    evaluation_interval=10,
    evaluation_episodes=3,
    checkpoint_path="checkpoints/cartpole_muzero/cartpole_state_v15.pt",
    reuse_checkpoint=False,
    seed=0,
    device=None,
):
    configuration = dict(locals())
    configuration.pop("device")
    validate_training_arguments(**configuration)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_state_env(env_id)
    if not hasattr(env.action_space, "n"):
        env.close()
        raise ValueError(f"{env_id} must have a discrete action space")
    action_dim = int(env.action_space.n)
    env.action_space.seed(seed)

    resume_data = None
    if reuse_checkpoint:
        candidates = [
            companion_checkpoint_path(checkpoint_path, "latest"),
            companion_checkpoint_path(checkpoint_path, "final"),
            Path(checkpoint_path),
        ]
        resume_path = next((path for path in candidates if path.exists()), None)
        if resume_path is None:
            env.close()
            attempted = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"--reuse-checkpoint found no checkpoint; tried: {attempted}"
            )
        try:
            representation, dynamics, prediction, resume_data = load_latent_checkpoint(
                resume_path,
                device,
            )
        except Exception:
            env.close()
            raise
        required = {"optimizer_state_dict", "replay_state", "completed_episodes"}
        missing = required - resume_data.keys()
        if missing:
            env.close()
            raise ValueError(
                f"checkpoint {resume_path} cannot resume training; missing state: "
                f"{', '.join(sorted(missing))}"
            )
        expected_architecture = (4, latent_dim, hidden_dim, hidden_dim, hidden_dim, action_dim)
        checkpoint_architecture = (
            representation.input_dim, representation.latent_dim,
            representation.hidden_dim, dynamics.hidden_dim,
            prediction.hidden_dim, prediction.action_dim,
        )
        if resume_data.get("training_config", {}).get("env_id") != env_id:
            env.close()
            raise ValueError("checkpoint environment does not match requested environment")
        if checkpoint_architecture != expected_architecture:
            env.close()
            raise ValueError(
                "checkpoint architecture/environment does not match requested "
                f"configuration: checkpoint={checkpoint_architecture}, "
                f"requested={expected_architecture}"
            )
        previous_config = resume_data.get("training_config", {})
        for immutable_name in ("discount", "reward_scale"):
            previous = previous_config.get(immutable_name)
            if previous is not None and not np.isclose(
                previous,
                configuration[immutable_name],
            ):
                env.close()
                raise ValueError(
                    f"cannot change {immutable_name} when reusing replay targets "
                    f"({previous} -> {configuration[immutable_name]})"
                )
    else:
        representation = RepresentationNetwork(latent_dim, hidden_dim).to(device)
        dynamics = DynamicsModel(latent_dim, action_dim, hidden_dim).to(device)
        prediction = PredictionNetwork(latent_dim, action_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        [
            *representation.parameters(),
            *dynamics.parameters(),
            *prediction.parameters(),
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    replay = LatentReplayBuffer(buffer_capacity)

    if resume_data is not None:
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        # Requested optimizer settings intentionally override saved param groups.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
            parameter_group["weight_decay"] = weight_decay
        replay.load_state_dict(resume_data["replay_state"])
        replay.resize(buffer_capacity)
        history = resume_data.get(
            "history",
            {"scores": [], "losses": [], "evaluations": []},
        )
        best = float(resume_data.get("best_score", -float("inf")))
        completed_episodes = int(resume_data["completed_episodes"])
        if "python_rng_state" in resume_data:
            random.setstate(resume_data["python_rng_state"])
        if "numpy_rng_state" in resume_data:
            np.random.set_state(resume_data["numpy_rng_state"])
        if "torch_rng_state" in resume_data:
            torch.set_rng_state(resume_data["torch_rng_state"].cpu())
        print(
            f"resuming from {resume_path}: completed={completed_episodes} "
            f"replay={len(replay)}",
            flush=True,
        )
    else:
        history = {"scores": [], "losses": [], "evaluations": []}
        best = -float("inf")
        completed_episodes = 0

    mcts = make_latent_mcts(
        dynamics,
        prediction,
        action_dim,
        simulations=simulations,
        discount=discount,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )
    update_mode = (
        f"fixed({updates_per_episode}/episode)"
        if updates_per_transition <= 0
        else (
            f"transition-relative(rate={updates_per_transition:g}, "
            f"min={min_updates_per_episode}, max={max_updates_per_episode})"
        )
    )
    training_config = {
        **configuration,
        "action_dim": action_dim,
        "update_mode": update_mode,
    }

    def training_metadata(completed):
        return {
            "history": history,
            "raw_episode_scores": history["scores"],
            "best_score": best,
            "discount": discount,
            "reward_scale": reward_scale,
            "update_mode": update_mode,
            "training_config": training_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_state": replay.state_dict(),
            "completed_episodes": completed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }

    final_episode = completed_episodes + episodes
    print(
        f"training {env_id}: episodes={completed_episodes + 1}-{final_episode} "
        f"simulations={simulations} device={device}",
        flush=True,
    )
    print(f"learner updates: {update_mode}", flush=True)

    try:
        for episode_index in range(completed_episodes, final_episode):
            observation, _ = env.reset(seed=seed + episode_index)
            env.action_space.seed(seed + episode_index)
            trajectory = []
            is_warmup = episode_index < warmup_episodes
            temperature = temperature_for_episode(
                episode_index,
                initial=temperature_initial,
                middle=temperature_middle,
                final=temperature_final,
                initial_episodes=temperature_initial_episodes,
                middle_episodes=temperature_middle_episodes,
            )

            for step in range(max_steps):
                latent = representation.encode(observation)
                root = mcts.search(latent, add_exploration_noise=True)
                if is_warmup:
                    action = env.action_space.sample()
                    policy = visit_count_policy(root, temperature=1.0)
                else:
                    action, policy = select_training_action(root, temperature)
                next_observation, raw_reward, terminated, truncated, _ = env.step(action)
                truncated = bool(truncated or (step + 1 == max_steps and not terminated))
                trajectory.append(
                    {
                        "observation": observation,
                        "action": action,
                        "reward": float(raw_reward),
                        "policy": policy,
                        "policy_valid": not is_warmup,
                        "value": root.mean_value,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break

            targets = bootstrapped_value_targets(
                trajectory,
                discount=discount,
                bootstrap_steps=bootstrap_steps,
                reward_scale=reward_scale,
            )
            replay_episode = []
            for transition, target in zip(trajectory, targets):
                replay_transition = dict(transition)
                replay_transition["reward"] *= reward_scale
                replay_transition["value"] = target
                replay_episode.append(replay_transition)

            raw_episode_score = sum(item["reward"] for item in trajectory)
            replay.add_episode(replay_episode)
            history["scores"].append(raw_episode_score)
            latest_loss = None
            update_count = updates_for_episode(
                len(trajectory),
                updates_per_episode,
                updates_per_transition,
                min_updates_per_episode,
                max_updates_per_episode,
            )
            for _ in range(update_count):
                latest_loss = train_latent_step(
                    representation,
                    dynamics,
                    prediction,
                    replay,
                    optimizer,
                    min(batch_size, len(replay)),
                    unroll_steps,
                    action_dim,
                    device,
                    gradient_clip_norm=gradient_clip_norm,
                    dynamics_gradient_scale=dynamics_gradient_scale,
                    policy_loss_weight=policy_loss_weight,
                    value_loss_weight=value_loss_weight,
                    reward_loss_weight=reward_loss_weight,
                )
                history["losses"].append(latest_loss)

            evaluation_text = ""
            if (
                (episode_index + 1) % evaluation_interval == 0
                or episode_index + 1 == final_episode
            ):
                evaluation_score, _ = evaluate(
                    env_id,
                    representation,
                    prediction,
                    mcts,
                    evaluation_episodes,
                    max_steps,
                    [seed + 10000 + i for i in range(evaluation_episodes)],
                )
                history["evaluations"].append(
                    (episode_index + 1, evaluation_score)
                )
                if evaluation_score > best:
                    best = evaluation_score
                    save_latent_checkpoint(
                        checkpoint_path,
                        representation,
                        dynamics,
                        prediction,
                        training_metadata(episode_index + 1),
                    )
                save_latent_checkpoint(
                    companion_checkpoint_path(checkpoint_path, "latest"),
                    representation,
                    dynamics,
                    prediction,
                    training_metadata(episode_index + 1),
                )
                evaluation_text = f" eval={evaluation_score:.1f} best={best:.1f}"

            phase_text = " warmup(policy-masked)" if is_warmup else ""
            loss_text = (
                f"{phase_text} updates=0"
                if latest_loss is None
                else (
                    f"{phase_text} updates={update_count:2d}"
                    f" loss={latest_loss['total_loss']:.3f}"
                    f" p={latest_loss['policy_loss']:.3f}"
                    f" v={latest_loss['value_loss']:.3f}"
                    f" r={latest_loss['reward_loss']:.3f}"
                )
            )
            print(
                f"episode {episode_index + 1:4d}/{final_episode} "
                f"score={raw_episode_score:6.1f} steps={len(trajectory):3d} "
                f"replay={len(replay):5d} temp={temperature:.2f}"
                f"{loss_text}{evaluation_text}",
                flush=True,
            )
    finally:
        env.close()

    final_metadata = training_metadata(final_episode)
    save_latent_checkpoint(
        companion_checkpoint_path(checkpoint_path, "final"),
        representation,
        dynamics,
        prediction,
        final_metadata,
    )
    save_latent_checkpoint(
        companion_checkpoint_path(checkpoint_path, "latest"),
        representation,
        dynamics,
        prediction,
        final_metadata,
    )
    reward_plot = save_reward_plot(
        history,
        Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_rewards.png"),
    )
    print(f"saved reward plot: {reward_plot}", flush=True)
    return representation, dynamics, prediction, history


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train latent MuZero jointly from CartPole native 4D observations."
    )
    parser.add_argument("--env", dest="env_id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--warmup-episodes", type=int, default=1)
    parser.add_argument("--updates-per-episode", type=int, default=5)
    parser.add_argument("--updates-per-transition", type=float, default=0.0)
    parser.add_argument("--min-updates-per-episode", type=int, default=1)
    parser.add_argument("--max-updates-per-episode", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--dynamics-gradient-scale", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--bootstrap-steps", type=int, default=10)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--pb-c-base", type=float, default=19652)
    parser.add_argument("--pb-c-init", type=float, default=1.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.25)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-initial", type=float, default=1.0)
    parser.add_argument("--temperature-middle", type=float, default=0.5)
    parser.add_argument("--temperature-final", type=float, default=0.25)
    parser.add_argument("--temperature-initial-episodes", type=int, default=200)
    parser.add_argument("--temperature-middle-episodes", type=int, default=500)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/cartpole_muzero/cartpole_state_v15.pt",
    )
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="continue latest state while preserving optimizer/replay/history",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    arguments = vars(build_argument_parser().parse_args())
    _, _, _, history = train_latent_muzero(**arguments)
    print(
        f"training complete: episodes={len(history['scores'])} "
        f"last_score={history['scores'][-1]:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
