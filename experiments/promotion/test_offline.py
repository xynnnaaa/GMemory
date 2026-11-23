"""No model, API credentials or ALFWorld assets required."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import sequences
from .policy import Gate


class PromotionTest(unittest.TestCase):
    def test_manifest(self):
        games = [f'{skill}-instance{index}/trial/game.tw-pddl' for skill in ('heat', 'cool', 'clean')
                 for index in range(8)]
        plans = list(sequences(games, 100, 42))
        self.assertEqual(plans, list(sequences(reversed(games), 100, 42)))
        self.assertEqual(sum(plan['kind'] == 'transfer' for plan in plans), 50)
        for plan in plans:
            names = plan['games']
            self.assertEqual(len(set(names)), 5)
            if plan['kind'] == 'transfer':
                self.assertEqual(names[0].split('-')[0], names[3].split('-')[0])

    def test_credit_and_freeze(self):
        item = SimpleNamespace(vector=[1., 0.], uses=1, successes=1, reward_sum=1.)
        def fit(rewards):
            gate = Gate(42)
            gate.decide(item, 0., [], 1, True)
            gate.decide(item, 0., [], 4, True)
            count, _ = gate.update(rewards, .03)
            self.assertEqual(count, 1)
            return gate
        only_past = fit([1, 1, 0, 0, 0])
        self.assertFalse(np.any(only_past.weights))
        future = fit([0, 0, 1, 0, 0])
        self.assertTrue(np.any(future.weights))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            future.save(path)
            frozen = Gate(123)
            frozen.load(path)
            decisions = [frozen.decide(item, 0., [], step, False) for step in range(134)]
            self.assertEqual(len(set(decisions)), 1)
            self.assertFalse(frozen.events)
            np.testing.assert_array_equal(future.weights, frozen.weights)
