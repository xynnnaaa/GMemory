"""ALFWorld/MacNet promotion-only RL experiment (run with python -m)."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

from .data import settings, inventory
from .policy import Gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'dev', 'test'])
    parser.add_argument('--method', choices=['learned', 'comem', 'first-use'], default='learned')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.03)
    args = parser.parse_args()
    training = args.mode == 'train'
    if (args.mode != 'test') != bool(args.manifest):
        parser.error('Train/dev require a manifest; test uses only the original benchmark')
    if training and (args.method != 'learned' or args.checkpoint or not 0 < args.lr <= 1):
        parser.error('Train a fresh learned gate with 0 < lr <= 1')
    if not training and (args.method == 'learned') != bool(args.checkpoint):
        parser.error('Only learned evaluation requires a checkpoint')

    sys.path.insert(0, str(Path('tasks').resolve()))
    import run as host
    from envs.alfworld_env import get_env_name_from_gamefile, prefixes
    from mas.memory import CoMem

    gate = Gate(args.seed)
    if args.checkpoint:
        gate.load(args.checkpoint)

    class Audited(CoMem):
        def init_task_context(self, *args, **kwargs):
            self.cross_reads = 0
            return super().init_task_context(*args, **kwargs)

        def save_task_context(self, *args, **kwargs):
            self.cross_reads = sum(owner != self.entries[index].owner
                                   for index, owners in self.used.items() if self.entries[index].promoted
                                   for owner in owners)
            return super().save_task_context(*args, **kwargs)

    class Learned(Audited):
        def should_promote(self, item, baseline):
            shared = (entry for entry in self.entries if entry.promoted)
            return gate.decide(item, baseline, shared, self.episodes, training)

    class FirstUse(Audited):
        def should_promote(self, item, baseline):
            return True

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {key: str(value) for key, value in vars(args).items()}
    for key in ('manifest', 'checkpoint'):
        path = getattr(args, key)
        if path:
            metadata[key + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2))
    if args.mode == 'test':
        sequences = [host.get_task('alfworld')]
        if len(sequences[0]) != 134:
            raise ValueError('Expected the original 134-task ALFWorld evaluation manifest')
        import yaml
        config = yaml.safe_load(Path(host.CONFIG['alfworld']['env_config_path']).read_text())
        env = host.get_env('alfworld', config, 30)
    else:
        split = 'train' if training else 'valid_seen'
        plan = json.loads(args.manifest.read_text())
        if plan['split'] != split or not plan['sequences']:
            raise ValueError('Empty or wrong-split manifest')
        config, root = settings(split)
        env = host.get_env('alfworld', config, 30)
        games = inventory(env.main_env, root)
        sequences = [row['games'] for row in plan['sequences']]
        for names in sequences:
            if len(names) != 5 or len(set(names)) != 5 or any(name not in games for name in names):
                raise ValueError('Expected five distinct official games from the requested split')

    for index, sequence in enumerate(sequences):
        random.seed(args.seed)
        np.random.seed(args.seed)
        directory = args.output / f'stream-{index:04d}'
        recorder = host.get_recorder('alfworld', str(directory), f'stream-{index:04d}')
        manager = host.TaskManager('alfworld', 'macnet', args.method, [], env, recorder,
                                   host.get_mas('macnet'), dict(host.CONFIG.get('macnet', {}), failed_topk=0),
                                   {'working_dir': str(directory), 'dataset': 'alfworld', 'hop': 1})
        host.build_mas(manager, 'io', 'comem', args.model,
                       memory_class={'learned': Learned, 'first-use': FirstUse, 'comem': Audited}[args.method])
        rewards = []
        cross_reads = []
        recorder.dataset_begin()
        for step, entry in enumerate(sequence):
            if args.mode == 'test':
                task = dict(entry)
            else:
                path = games[entry]
                env.main_env.game_files = [path]
                probe = env.main_env.init_env(batch_size=1)
                try:
                    observations, _ = probe.reset()
                    goal = observations[0].strip()
                    if 'You are in the middle of a room.' in goal:
                        goal = goal[goal.index('You are in the middle of a room.'):]
                    if 'Your task is to:' not in goal:
                        raise ValueError(f'Missing task instruction in {entry}')
                finally:
                    probe.close()
                name = get_env_name_from_gamefile(entry)
                task = {'task': goal, 'env_name': name, 'task_type': prefixes[name],
                        'env_kwargs': {'config': 'alfworld', 'gamefile': path}}
            reward, _ = host.run_episode(manager, step, task)
            rewards.append(float(reward))
            cross_reads.append(manager.mas.meta_memory.cross_reads)
        recorder.dataset_end()
        metrics = {'stream': index, 'rewards': rewards, 'cross_reads': cross_reads,
                   'success_rate': sum(rewards) / len(rewards)}
        if training:
            metrics['eligible'], metrics['promotions'] = gate.update(rewards, args.lr)
            if gate.weights is not None:
                gate.save(args.output / 'policy.json')
                if (index + 1) % 20 == 0:
                    gate.save(args.output / f'policy-{index+1}.json')
        with (args.output / 'metrics.jsonl').open('a') as output:
            print(json.dumps(metrics), file=output)
        for handler in list(recorder.logger.handlers):
            recorder.logger.removeHandler(handler)
            handler.close()
    if training and gate.weights is None:
        raise RuntimeError('No eligible decisions across training streams')


if __name__ == '__main__':
    main()
