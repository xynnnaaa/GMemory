"""Paper-based CoMem reproduction; see README for underspecified choices."""

import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from mas.llm import Message
from .memory_base import MASMemoryBase


@dataclass
class Insight:
    owner: str
    text: str
    vector: list
    score: float = 50.0
    uses: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    misses: int = 0
    promoted: bool = False


def unit(vector):
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('CoMem requires a finite, nonzero embedding')
    return (values / norm).tolist()


class CoMem(MASMemoryBase):
    def __post_init__(self):
        super().__post_init__()
        self.path = Path(self.persist_dir) / 'comem.json'
        state = json.loads(self.path.read_text()) if self.path.exists() else {
            'episodes': 0, 'total_reward': 0, 'entries': []}
        self.entries = [Insight(**item) for item in state['entries']]
        self.episodes = state['episodes']
        self.total_reward = state['total_reward']
        self.window = 6 if self.global_config.get('dataset') == 'alfworld' else 5
        self.owners = set()

    def bind(self, agents):
        for agent in agents:
            self.owners.add(agent.name)
            agent.reasoning = partial(self._reason, agent.name, agent.profile, agent.reasoning)

    def init_task_context(self, task_main, task_description=None):
        self.traces = defaultdict(list)
        self.steps = []
        self.used = defaultdict(set)
        self.selected = {}
        return super().init_task_context(task_main, task_description)

    def _collective(self, ranked, depth):
        candidates = ranked[:4 * depth]
        clusters = [[index] for index in candidates]
        distances = {(first, second): 1 - np.clip(np.dot(self.entries[first].vector,
                     self.entries[second].vector), -1, 1) for first in candidates for second in candidates}
        while len(clusters) > 1:
            distance, left, right = min(
                (float(np.mean([distances[first, second] for first in clusters[left]
                                for second in clusters[right]])), left, right)
                for left in range(len(clusters)) for right in range(left + 1, len(clusters)))
            if distance > 0.2:
                break
            clusters[left].extend(clusters.pop(right))
        medoids = [min(group, key=lambda index: (sum(distances[index, other] for other in group),
                                                 candidates.index(index))) for group in clusters]
        return sorted(medoids, key=candidates.index)[:depth]

    def retrieve_memory(self, query_task, insight_topk=3, **kwargs):
        depth = max(0, insight_topk)
        if self.entries and depth:
            query = unit(self.embedding_func.embed_query(query_task))
            ranked = sorted(range(len(self.entries)), reverse=True,
                            key=lambda index: np.dot(query, self.entries[index].vector))
            with ThreadPoolExecutor(max_workers=1) as executor:
                shared = executor.submit(self._collective, [index for index in ranked
                                         if self.entries[index].promoted], depth)
                private = {owner: [index for index in ranked if not self.entries[index].promoted
                                   and self.entries[index].owner == owner][:depth] for owner in self.owners}
                collective = shared.result()
            self.selected = {owner: personal + collective for owner, personal in private.items()}
        return [], [], []

    def _reason(self, owner, role, reasoning, messages, config):
        selected = self.selected.get(owner, [])
        if selected:
            guidance = '\n'.join(('[Collective] ' if self.entries[index].promoted else '[Private] ')
                                 + self.entries[index].text for index in selected)
            messages = [messages[0], Message('user', 'Experience for this task:\n' + guidance +
                        '\n\n' + messages[1].content), *messages[2:]]
        answer = reasoning(messages, config)
        self.traces[owner].append({'step': len(self.steps), 'role': role, 'response': answer})
        for index in selected:
            self.used[index].add(owner)
        return answer

    def move_memory_state(self, action, observation, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        self.steps.append({'action': action, 'observation': observation})

    def _ask(self, instruction, evidence):
        raw = self.llm_model([Message('system', instruction), Message('user', json.dumps(evidence))],
                             temperature=0)
        body = (raw or '').strip()
        decoder = json.JSONDecoder()
        try:
            return decoder.raw_decode(body)[0]
        except json.JSONDecodeError:
            fence = re.search(r'^[ \t]*```(?:json)?[ \t]*\r?\n', body, re.IGNORECASE | re.MULTILINE)
            if fence:
                return decoder.raw_decode(body[fence.end():].lstrip())[0]
            raise

    def save_task_context(self, label, feedback=None, *, reward):
        reward = float(reward)
        if not 0 <= reward <= 1:
            raise ValueError('Expected normalized final environment reward in [0, 1]')
        context = self.current_task_context
        evidence = {owner: {'agent': owner, 'task': context.task_main, 'initial_state': context.task_description,
                    'own_decisions': trace, 'environment_steps': self.steps, 'reward': reward,
                    'feedback': feedback} for owner, trace in self.traces.items()}
        baseline = self.total_reward / self.episodes if self.episodes else reward
        updated = []
        for index, original in enumerate(self.entries):
            item = replace(original)
            users = self.used.get(index)
            if users:
                item.uses += 1
                item.successes += int(label)
                item.reward_sum += reward
                item.misses = 0
                if item.promoted:
                    scores = [self._ask('Act as the indicated agent. Rate how useful the insight was for '
                              'your decisions in this episode. Return only a JSON number from 0 to 10.',
                              dict(evidence[owner], insight=item.text)) for owner in sorted(users)]
                    if not all(type(score) in (int, float) and 0 <= score <= 10 for score in scores):
                        raise ValueError('Invalid CoMem utility rating')
                    utility = 0.5 * reward + 0.5 * sum(scores) / (10 * len(scores))
                    item.score = 0.5 * item.score + 0.5 * 100 * utility
                    if item.score < 30:
                        continue
                elif item.uses >= 3 and (item.reward_sum / item.uses >= baseline or
                                         math.isclose(item.reward_sum / item.uses, baseline, abs_tol=1e-12)):
                    item.promoted = True
            elif not item.promoted:
                item.misses += 1
                if item.misses > self.window:
                    continue
            updated.append(item)
        for owner, trace in evidence.items():
            texts = self._ask('Reflect only on the indicated agent’s own decisions and observed outcomes. '
                              'Distill up to three short, transferable, actionable personal insights. '
                              'Return only a JSON array of strings; return [] if none are warranted.', trace)
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError('Invalid CoMem reflection')
            texts = list(dict.fromkeys(text.strip() for text in texts if text.strip()))[:3]
            vectors = self.embedding_func.embed_documents(texts) if texts else []
            if len(vectors) != len(texts):
                raise ValueError('Missing CoMem embeddings')
            updated.extend(Insight(owner, text, unit(vector)) for text, vector in zip(texts, vectors))
        state = {'episodes': self.episodes + 1, 'total_reward': self.total_reward + reward,
                 'entries': [asdict(item) for item in updated]}
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, allow_nan=False), encoding='utf-8')
        os.replace(temporary, self.path)
        self.entries, self.episodes, self.total_reward = updated, state['episodes'], state['total_reward']
        context.label = label
        return context
