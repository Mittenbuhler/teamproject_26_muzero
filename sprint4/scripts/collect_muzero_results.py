"""Rebuild saved-run figures, evaluations and six-second GIFs without training.

Run with the project environment: python -m scripts.collect_muzero_results
Use --plots-only to redraw figures from existing result JSON without evaluation.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'muzero_results_matplotlib'))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(20000, 20020))
METHODS = ['random', 'raw_policy', 'mcts']
LABELS = ['Random actions', 'Trained policy without search', 'MuZero with search']
RUNS = {
    'cartpole': dict(package='cartpole_muzero', run='state4d_seed0_R9ncWC',
                     checkpoint='run.pt', history='run_latest.pt',
                     title='MuZero on CartPole',
                     input='Input: cart position, cart velocity, pole angle and angular velocity',
                     ylabel='CartPole reward (maximum 500)', ylim=(0, 550), ticks=list(range(0, 501, 100))),
    'breakout': dict(package='minatar_muzero', run='breakout_screenshots',
                     checkpoint='best.pt', history='latest.pt',
                     title='MuZero on MinAtar Breakout', input='Input: four 32 x 32 RGB screenshots',
                     ylabel='Breakout score', ylim=(0, 7.5), ticks=list(range(8))),
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def file_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest.hexdigest())


def summarize(scores, cap=None):
    a = np.asarray(scores, dtype=float)
    result = dict(n=len(a), mean=float(a.mean()),
                  sample_std=float(a.std(ddof=1)) if len(a) > 1 else None,
                  median=float(np.median(a)), min=float(a.min()), max=float(a.max()), scores=scores)
    if cap is not None:
        result['episodes_at_limit'] = int((a == cap).sum())
    return result


def save_gif(frames, durations, path):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    with Image.open(path) as gif:
        actual = []
        for i in range(gif.n_frames):
            gif.seek(i)
            actual.append(gif.info['duration'])
        assert sum(actual) == 6000
        return dict(frames=gif.n_frames, duration_ms=sum(actual), dimensions=list(gif.size),
                    frame_durations_ms=actual)


def cartpole_gif(h, f, mcts, cfg, report, path):
    import torch
    from cartpole_muzero.environment import make_state_env
    from cartpole_muzero.mcts import select_action
    successful = [seed for seed, score in zip(SEEDS, report['evaluation']['mcts']['scores'])
                  if score == cfg['max_steps']]
    seed = successful[0] if successful else SEEDS[0]
    expected_steps = int(report['evaluation']['mcts']['scores'][SEEDS.index(seed)])
    assert expected_steps > 200, 'A split at step 200 requires a longer recorded episode.'
    first = np.rint(np.linspace(0, 200, 60)).astype(int).tolist()
    last = np.rint(np.linspace(201, expected_steps, 60)).astype(int).tolist()
    selected = first + last
    wanted = set(selected)
    fonts = {weight: font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight=weight))
             for weight in ['normal', 'bold']}
    title_font = ImageFont.truetype(fonts['bold'], 23)
    small_font = ImageFont.truetype(fonts['normal'], 15)
    label_font = ImageFont.truetype(fonts['bold'], 16)
    env = make_state_env(cfg['env_id'], render_mode='rgb_array')
    frames = {}
    def capture(step):
        play = Image.fromarray(env.render()).convert('RGB')
        image = Image.new('RGB', (play.width, play.height + 92), 'white')
        image.paste(play, (0, 62))
        draw = ImageDraw.Draw(image)
        draw.text((20, 10), 'MuZero on CartPole', font=title_font, fill='#172d3a')
        draw.text((20, 39), 'Successful example episode' if successful else 'Example episode',
                  font=small_font, fill='#536673')
        phase = 'First 200 steps' if step <= 200 else f'Remaining {expected_steps - 200} steps'
        draw.text((20, image.height - 26), phase, font=label_font, fill='#167b80')
        label = f'Step {step} / {expected_steps}'
        draw.text((image.width - 20 - draw.textlength(label, font=label_font), image.height - 26),
                  label, font=label_font, fill='#172d3a')
        return image
    np.random.seed(seed)
    score = 0.0
    try:
        env.action_space.seed(seed)
        observation, _ = env.reset(seed=seed)
        frames[0] = capture(0)
        with torch.inference_mode():
            for step in range(1, cfg['max_steps'] + 1):
                root = mcts.search(h.encode(observation), add_exploration_noise=False)
                action = select_action(root, temperature=0.0)[0]
                observation, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                if step in wanted:
                    frames[step] = capture(step)
                if terminated or truncated:
                    break
    finally:
        env.close()
    assert step == expected_steps and score == expected_steps
    result = save_gif([frames[s] for s in selected], [50] * 120, path)
    assert sum(result['frame_durations_ms'][:60]) == 3000
    result.update(seed=seed, score=score, steps=step,
                  seed_selection='First 500-step success in the existing evaluation seed list; an illustrative successful episode.'
                  if successful else 'First evaluation seed.',
                  segments=[dict(seconds=[0, 3], steps=[0, 200], sampled_steps=first),
                            dict(seconds=[3, 6], steps=[201, expected_steps], sampled_steps=last)])
    return result


def breakout_gif(h, g, f, mcts, data, cfg, path):
    from minatar_muzero.diagnose import record_episode_gif
    with tempfile.TemporaryDirectory(prefix='breakout_recording_') as directory:
        original = Path(directory) / 'recording.gif'
        result = record_episode_gif(original, data['game'], h, f, mcts,
                                    data['base_observation_shape'], data['history_length'],
                                    seed=SEEDS[0], max_steps=cfg['max_steps'], fps=10, output_width=480,
                                    sticky_action_prob=data['sticky_action_prob'],
                                    difficulty_ramping=data['difficulty_ramping'])
        with Image.open(original) as source:
            frames = []
            for i in range(source.n_frames):
                source.seek(i)
                frames.append(source.convert('RGB').copy())
        boundaries = [round(i * 600 / len(frames)) * 10 for i in range(len(frames) + 1)]
        result.update(save_gif(frames, [b - a for a, b in zip(boundaries, boundaries[1:])], path))
    result.update(seed_selection='First evaluation seed; not selected by reward.',
                  timing_note='Full episode uniformly retimed to six seconds. This episode ends before step 200.')
    result.pop('path', None)
    return result


def collect(name, info, out):
    import torch
    import gymnasium
    torch.set_num_threads(1)
    random.seed(SEEDS[0]); np.random.seed(SEEDS[0]); torch.manual_seed(SEEDS[0])
    folder = ROOT / 'checkpoints' / info['package'] / info['run']
    checkpoint, history_path = folder / info['checkpoint'], folder / info['history']
    print(f'{name}: extracting saved training history', flush=True)
    saved = torch.load(history_path, map_location='cpu', weights_only=False)
    history = dict(source=file_identity(history_path), completed_episodes=saved['completed_episodes'],
                   scores=saved['history']['scores'], evaluations=saved['history']['evaluations'],
                   learner_updates=len(saved['history']['losses']), training_config=saved['training_config'])
    write_json(out / 'training_history.json', history)
    del saved; gc.collect()
    with (out / 'training_scores.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['episode', 'actor_reward'])
        writer.writerows(enumerate(history['scores'], 1))
    with (out / 'periodic_evaluations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['episode', 'mean_mcts_reward'])
        writer.writerows(history['evaluations'])
    if name == 'cartpole':
        from cartpole_muzero.train import load_latent_checkpoint, make_latent_mcts, evaluate
    else:
        from minatar_muzero.train import load_latent_checkpoint, make_latent_mcts, evaluate
    h, g, f, data = load_latent_checkpoint(checkpoint, torch.device('cpu'))
    # Evaluation needs weights and configuration, not the large screenshot
    # replay or optimizer state. Release those before rendering and writing.
    data.pop('replay_state', None)
    data.pop('optimizer_state_dict', None)
    gc.collect()
    for network in (h, g, f):
        network.eval()
    cfg = data['training_config']
    simulations = cfg['simulations']
    periodic_seed_count = cfg['evaluation_episodes']
    periodic_seeds = list(range(cfg['seed'] + 10000, cfg['seed'] + 10000 + periodic_seed_count))
    assert not set(SEEDS) & set(periodic_seeds)
    mcts = make_latent_mcts(g, f, data['action_dim'], simulations=simulations,
                            discount=data['discount'], pb_c_base=cfg['pb_c_base'], pb_c_init=cfg['pb_c_init'])
    report = dict(status='running', training_performed=False, checkpoint=file_identity(checkpoint),
                  completed_training_episodes=data['completed_episodes'],
                  best_scheduled_evaluation=data['best_score'],
                  settings=dict(seeds=SEEDS, episodes_per_method=len(SEEDS), simulations=simulations,
                                max_steps=cfg['max_steps'], root_exploration_noise=False,
                                periodic_evaluation_seeds=periodic_seeds),
                  versions=dict(torch=torch.__version__, numpy=np.__version__, gymnasium=gymnasium.__version__),
                  training_config=cfg, evaluation={},
                  limitations=['One trained model per environment, not multiple independent training runs.',
                               'Training actor scores use changing exploration settings.',
                               'CartPole and Breakout have different reward scales, inputs and search budgets.'])
    report['settings']['environment_lifecycle'] = (
        'One environment instance per method, reset with each seed in order.'
        if name == 'breakout' else 'New CartPole environment for each episode.')
    if name == 'breakout':
        report['limitations'].append('Installed MinAtar retains sticky-action memory across resets. Environment lifecycle and seed order are part of this original evaluation protocol.')
    for mode in METHODS:
        if name == 'breakout' and mode != 'random':
            with torch.inference_mode():
                _, scores = evaluate(data['game'], h, f, mcts, len(SEEDS), cfg['max_steps'], data['history_length'],
                                      SEEDS, mode == 'mcts', sticky_action_prob=data['sticky_action_prob'],
                                      difficulty_ramping=data['difficulty_ramping'])
            report['evaluation'][mode] = summarize([float(score) for score in scores])
            write_json(out / 'evaluation.json', report)
            print(f'{name} {mode}: 20/20, mean={np.mean(scores):.2f}', flush=True)
            continue
        scores = []
        env = None
        if name == 'breakout':
            from minatar_muzero.environment import make_minatar_env
            env = make_minatar_env(data['game'], image_size=data['base_observation_shape'][1],
                                   sticky_action_prob=data['sticky_action_prob'],
                                   difficulty_ramping=data['difficulty_ramping'])
        try:
            for i, seed in enumerate(SEEDS, 1):
                with torch.inference_mode():
                    if name == 'cartpole':
                        _, values = evaluate(cfg['env_id'], h, f, mcts, 1, cfg['max_steps'], [seed],
                                             use_mcts=mode == 'mcts', random_actions=mode == 'random')
                        score = values[0]
                    else:
                        env.reset(seed=seed); env.action_space.seed(seed); score = 0.0
                        for _ in range(cfg['max_steps']):
                            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                            score += float(reward)
                            if terminated or truncated:
                                break
                scores.append(float(score))
                report['evaluation'][mode] = summarize(scores, 500 if name == 'cartpole' else None)
                write_json(out / 'evaluation.json', report)
                if mode == 'mcts' or i == len(SEEDS):
                    print(f'{name} {mode}: {i}/20, score={score:g}', flush=True)
        finally:
            if env is not None:
                env.close()
    raw = np.array(report['evaluation']['raw_policy']['scores'])
    searched = np.array(report['evaluation']['mcts']['scores'])
    delta = searched - raw
    report['search_vs_policy'] = dict(mean_difference=float(delta.mean()),
                                       relative_mean_improvement_percent=float(100 * delta.mean() / raw.mean()),
                                       wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                                       paired_differences=delta.tolist())
    with (out / 'episode_scores.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['seed', *METHODS, 'mcts_minus_raw_policy'])
        for i, seed in enumerate(SEEDS):
            writer.writerow([seed, *[report['evaluation'][m]['scores'][i] for m in METHODS], delta[i]])
    print(f'{name}: recording six-second gameplay GIF', flush=True)
    gif_path = out / 'gameplay_6s.gif'
    gif = cartpole_gif(h, f, mcts, cfg, report, gif_path) if name == 'cartpole' else breakout_gif(h, g, f, mcts, data, cfg, gif_path)
    gif.update(path=str(gif_path.relative_to(ROOT)), checkpoint=report['checkpoint'], simulations=simulations,
               root_exploration_noise=False)
    assert gif['score'] == report['evaluation']['mcts']['scores'][SEEDS.index(gif['seed'])]
    write_json(out / 'gameplay.json', gif)
    assert file_identity(checkpoint) == report['checkpoint']
    assert file_identity(history_path) == history['source']
    report.update(status='complete', checkpoint_unchanged=True, history_checkpoint_unchanged=True)
    write_json(out / 'evaluation.json', report)
    print(f'{name}: saved results without changing checkpoints', flush=True)
    del h, g, f, data; gc.collect()


def plot_training(ax, info, history):
    scores = np.asarray(history['scores']); episodes = np.arange(1, len(scores) + 1)
    ax.plot(episodes, scores, color='#c7d8df', alpha=.75, lw=.65, label='Training episodes')
    ax.plot(episodes[49:], np.convolve(scores, np.ones(50) / 50, mode='valid'),
            color='#167b80', lw=2.2, label='50-episode mean')
    ax.plot(*zip(*history['evaluations']), color='#c25836', lw=1.6, label='Periodic MCTS evaluation')
    ax.set(xlabel='Training episode', ylabel=info['ylabel'], ylim=info['ylim'], xlim=(0, len(scores)))
    ax.set_yticks(info['ticks'])
    ax.set_xticks(np.arange(0, len(scores) + 1, 100 if len(scores) == 800 else 300))
    ax.legend(loc='upper left', fontsize=10.5, facecolor='white', edgecolor='white', framealpha=.92)
    ax.grid(axis='y', color='#e4eaee', lw=.7); ax.set_axisbelow(True)


def plot_comparison(ax, info, report):
    values = [report['evaluation'][m]['mean'] for m in METHODS]
    bars = ax.bar(range(3), values, width=.66, color=['#a6b0b9', '#46a4a3', '#167b80'], zorder=2)
    for i, mode in enumerate(METHODS):
        ax.scatter(i + np.linspace(-.17, .17, 20), report['evaluation'][mode]['scores'], s=14, color='#233443', zorder=3)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=7, fontsize=13, fontweight='bold',
                 zorder=5, bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=1.5))
    ax.set_xticks(range(3), ['Random\nactions', 'Policy\n(no search)', 'MuZero\nwith search'])
    ax.tick_params(axis='x', labelsize=11, length=0, pad=10)
    ax.set(ylabel=info['ylabel'], ylim=info['ylim']); ax.set_yticks(info['ticks'])
    ax.grid(axis='y', color='#e4eaee', lw=.7); ax.set_axisbelow(True)


def render(name, info, out):
    history = json.loads((out / 'training_history.json').read_text())
    report = json.loads((out / 'evaluation.json').read_text())
    assert report['status'] == 'complete'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#172d3a', 'axes.labelcolor': '#243946',
                         'xtick.color': '#4d5f6c', 'ytick.color': '#4d5f6c', 'axes.edgecolor': '#667680'})
    for stem, drawing in [('training_history', lambda ax: plot_training(ax, info, history)),
                          ('performance_comparison', lambda ax: plot_comparison(ax, info, report))]:
        fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
        drawing(ax)
        ax.set_title(info['title'] + (' - training history' if stem == 'training_history' else ' - 20 fresh seeds'))
        for ext in ['png', 'svg']:
            fig.savefig(out / f'{stem}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.text(.052, .935, info['title'], fontsize=30, fontweight='bold', va='top')
    fig.text(.054, .875, info['input'], fontsize=16, color='#536673', va='top')
    fig.text(.066, .782, 'Training history', fontsize=18, fontweight='bold')
    fig.text(.661, .782, 'Performance on fresh seeds', fontsize=18, fontweight='bold')
    fixed = len(report['settings']['periodic_evaluation_seeds'])
    fig.text(.066, .750, f'{len(history["scores"]):,} episodes. Periodic evaluation uses {fixed} fixed seeds.', fontsize=11.5, color='#536673')
    fig.text(.661, .750, f'Checkpoint at episode {report["completed_training_episodes"]}. 20 episodes per method.', fontsize=11.5, color='#536673')
    plot_training(fig.add_axes([.070, .285, .510, .422]), info, history)
    plot_comparison(fig.add_axes([.669, .285, .286, .422]), info, report)
    scores = np.asarray(history['scores'])
    fig.text(.066, .169, 'Reward increases during training', fontsize=18, fontweight='bold')
    fig.text(.066, .124, f'First 100 episodes: {scores[:100].mean():.2f}. Last 100: {scores[-100:].mean():.2f}.', fontsize=13, color='#536673')
    delta = report['search_vs_policy']['mean_difference']
    fig.text(.661, .169, f'{delta:+.2f} average reward with search', fontsize=18, fontweight='bold', color='#167b80')
    fig.text(.661, .124, f'{report["evaluation"]["raw_policy"]["mean"]:.2f} without search vs. {report["evaluation"]["mcts"]["mean"]:.2f} with search', fontsize=13, color='#536673')
    fig.text(.054, .063, f'Evaluation: seeds 20000-20019, {report["settings"]["simulations"]} search simulations per move, no exploration noise. Dots show individual episodes.', fontsize=10.5, color='#536673')
    fig.text(.054, .032, 'One trained model per game. Training includes exploration. Source: saved training histories and fresh evaluations.', fontsize=10.5, color='#536673')
    fig.savefig(out / 'overview.png', dpi=200); plt.close(fig)
    with Image.open(out / 'overview.png') as image:
        assert image.size == (3200, 1800)
    comparison = report['search_vs_policy']
    lines = [f'# {info["title"]}: saved-run results', '',
             '[Combined figure](overview.png) | [Training history](training_history.png) | [Performance comparison](performance_comparison.png) | [Six-second gameplay GIF](gameplay_6s.gif)', '',
             f'Checkpoint: `{report["checkpoint"]["path"]}`, selected at episode {report["completed_training_episodes"]}.', '',
             f'Evaluation uses 20 fresh seeds (20000-20019), {report["settings"]["simulations"]} search simulations per move and no root exploration noise. Standard deviation is across environment seeds.', '',
             '| Method | Mean reward | Sample SD | Median | Range |' + (' Reached 500 |' if name == 'cartpole' else ''),
             '| --- | ---: | ---: | ---: | --- |' + (' ---: |' if name == 'cartpole' else '')]
    for mode, label in zip(METHODS, LABELS):
        s = report['evaluation'][mode]
        row = f'| {label} | {s["mean"]:.2f} | {s["sample_std"]:.2f} | {s["median"]:.2f} | {s["min"]:g}-{s["max"]:g} |'
        if name == 'cartpole':
            row += f' {s["episodes_at_limit"]}/20 |'
        lines.append(row)
    lines += ['', f'Search changes mean reward by **{delta:+.2f}**. Across matched seeds: {comparison["wins"]} wins, {comparison["ties"]} ties and {comparison["losses"]} losses.', '',
              'Additional search has almost no effect on this CartPole evaluation.' if name == 'cartpole' else 'Search improves average Breakout performance for this checkpoint.', '',
              '## Training history', '',
              f'The full history has {len(scores):,} episodes. Mean actor reward is {scores[:100].mean():.2f} in the first 100 episodes and {scores[-100:].mean():.2f} in the last 100. Exploration changes during training.', '',
              f'The best scheduled evaluation is {report["best_scheduled_evaluation"]:.2f}. The final scheduled evaluation is {history["evaluations"][-1][1]:.2f}. These use fixed seeds and are distinct from the fresh evaluation above.', '',
              '## Gameplay', '']
    gif = json.loads((out / 'gameplay.json').read_text())
    lines += [f'The GIF shows seed {gif["seed"]}, score {gif["score"]:g}, {gif["steps"]} environment steps in exactly six seconds.', '',
              'CartPole uses a successful example. The first three seconds show steps 0-200, and the final three show steps 201-500.' if name == 'cartpole' else 'Breakout uses the first evaluation seed. Its 62-step episode ends before step 200, so the full episode is uniformly retimed.', '',
              '## Data and reproduction', '',
              '- [Evaluation statistics and checkpoint identity](evaluation.json)',
              '- [Per-seed scores](episode_scores.csv)',
              '- [Full training history](training_history.json)',
              '- [Training rewards](training_scores.csv)',
              '- [Periodic evaluations](periodic_evaluations.csv)',
              '- [GIF timing and selection](gameplay.json)', '',
              f'Recreate with `python -m scripts.collect_muzero_results --model {name}` from sprint4. Use `--plots-only` to redraw existing results without running evaluation.', '',
              'The collector makes no optimizer updates and verifies checkpoint bytes are unchanged. These results concern one training run. Reward scales differ between environments.', '']
    if name == 'breakout':
        lines += ['## Reset protocol', '',
                  'This comparison reproduces the original procedure: one environment per method, with seeds 20000-20019 evaluated in order. The installed MinAtar version retains its previous sticky action across resets, so environment lifecycle is part of the protocol.', '']
        alternate = out / 'reset_protocol_check.json'
        if alternate.exists():
            check = json.loads(alternate.read_text())
            lines += [f'A [separate check with a newly constructed environment per seed](reset_protocol_check.json) gives policy-only mean {check["evaluation"]["raw_policy"]["mean"]:.2f} and MCTS mean {check["evaluation"]["mcts"]["mean"]:.2f}. The policy score differs on seed 20018. These are different reset procedures, not different model weights.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print('Ready:', out.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', *RUNS], default='all')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    names = list(RUNS) if args.model == 'all' else [args.model]
    for name in names:
        info = RUNS[name]
        out = ROOT / 'artifacts' / info['package'] / 'results' / info['run']
        out.mkdir(parents=True, exist_ok=True)
        if not args.plots_only:
            collect(name, info, out)
        render(name, info, out)


if __name__ == '__main__':
    main()
