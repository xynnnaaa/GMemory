"""Bernoulli promotion gate; no dependencies on the MAS or LLM client."""

import json
from pathlib import Path

import numpy as np


class Gate:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.baselines = np.zeros(4)
        self.updates = 0
        self.events = []

    def decide(self, item, baseline, shared, step, train):
        novelty = 1 - max((np.dot(item.vector, other.vector) for other in shared), default=0)
        features = np.r_[1., min(item.uses, 10) / 10, item.successes / item.uses,
                         item.reward_sum / item.uses, baseline, novelty, item.vector]
        if self.weights is None:
            if not train:
                raise ValueError('Evaluation requires trained gate weights')
            self.weights = np.zeros_like(features)
        prob = 1 / (1 + np.exp(-np.clip(self.weights @ features, -40, 40)))
        action = self.rng.random() < prob if train else prob >= 0.5
        if train and step < 4:
            self.events.append((step, (float(action) - prob) * features, bool(action)))
        return bool(action)

    def update(self, rewards, lr):
        if len(rewards) != 5:
            raise ValueError('Expected exactly five rewards')
        returns = np.array([sum(rewards[step + 1:]) for step in range(4)])
        if self.events:
            gradient = sum((returns[step] - self.baselines[step]) * score
                           for step, score, _ in self.events)
            self.weights += lr * gradient / max(1., np.linalg.norm(gradient))
        self.baselines += (returns - self.baselines) / (self.updates + 1)
        self.updates += 1
        counts = len(self.events), sum(action for _, _, action in self.events)
        self.events.clear()
        return counts

    def save(self, path):
        if self.weights is None:
            raise ValueError('No eligible private insight encountered; no policy to save')
        temp = Path(path).with_suffix('.tmp')
        temp.write_text(json.dumps({'version': 1, 'weights': self.weights.tolist()}))
        temp.replace(path)

    def load(self, path):
        state = json.loads(Path(path).read_text())
        if state['version'] != 1:
            raise ValueError('Unsupported promotion policy')
        self.weights = np.asarray(state['weights'], dtype=float)
        if self.weights.ndim != 1 or not np.isfinite(self.weights).all():
            raise ValueError('Invalid promotion weights')
