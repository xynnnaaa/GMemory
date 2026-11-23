"""Construct portable, split-relative five-game manifests from official games."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def sequences(files, count, seed):
    pool = sorted(set(files))
    if len(pool) < 5 or count < 1:
        raise ValueError('Need at least five distinct games and a positive sequence count')
    groups = defaultdict(list)
    for name in pool:
        groups[Path(name).parts[0].split('-')[0]].append(name)
    skills = sorted(key for key, values in groups.items() if len(values) >= 2)
    rng = random.Random(seed)
    for index in range(count):
        if index % 2 == 0:
            yield {'kind': 'random', 'games': rng.sample(pool, 5)}
        else:
            first, fourth = rng.sample(groups[rng.choice(skills)], 2)
            second, third, fifth = rng.sample([name for name in pool if name not in (first, fourth)], 3)
            yield {'kind': 'transfer', 'games': [first, second, third, fourth, fifth]}


def settings(split):
    import yaml

    config = yaml.safe_load(Path('tasks/env_configs/alfworld_config.yaml').read_text())
    config['split'] = {'train': 'train', 'valid_seen': 'eval_in_distribution'}[split]
    root = Path(config['dataset']['data_path' if split == 'train' else 'eval_id_data_path']).resolve()
    return config, root


def inventory(env, root):
    return {Path(name).resolve().relative_to(root).as_posix(): str(Path(name).resolve())
            for name in env.game_files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'valid_seen'], default='train')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from alfworld.agents import environment
    config, root = settings(args.split)
    official = getattr(environment, config['env']['type'])(config, train_eval=config['split'])
    plans = list(sequences(inventory(official, root), args.count, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump({'split': args.split, 'seed': args.seed, 'sequences': plans}, output, indent=2)


if __name__ == '__main__':
    main()
