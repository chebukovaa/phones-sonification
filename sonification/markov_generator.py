from __future__ import annotations

import numpy as np


class MarkovChainMelodyGenerator:
    """Markov chain over (pitch, duration) states."""

    def __init__(self, states: list[tuple[str, float]]):
        self.states = states  # набор нот, из которых составляется мелодия
        self.initial_probabilities = np.zeros(len(states))  # начальные вероятности
        self.transition_matrix = np.zeros((len(states), len(states)))  # матрица перехода
        self._idx = {s: i for i, s in enumerate(states)}  # уникальная индексация нот для обращения к ним

    def train(self, sequence: list[tuple[str, float]]) -> MarkovChainMelodyGenerator:
        for state in sequence:
            self.initial_probabilities[self._idx[state]] += 1
        total = self.initial_probabilities.sum()
        if total:
            self.initial_probabilities /= total

        for i in range(len(sequence) - 1):
            self.transition_matrix[self._idx[sequence[i]], self._idx[sequence[i + 1]]] += 1

        row_sums = self.transition_matrix.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition_matrix = np.where(
                row_sums[:, None], self.transition_matrix / row_sums[:, None], 0
            )
        return self

    def generate_phrase(self, n_notes: int, prev_note=None) -> list[tuple[str, float]]:
        melody = [prev_note if prev_note else self._random_note()]
        for _ in range(n_notes if prev_note else (n_notes - 1)):
            cur = melody[-1]
            row = self.transition_matrix[self._idx[cur]]
            if row.sum() > 0:
                melody.append(self.states[np.random.choice(len(self.states), p=row)])
            else:
                melody.append(self._random_note())
        return melody[1:] if prev_note else melody

    def _random_note(self) -> tuple[str, float]:
        return self.states[np.random.choice(len(self.states), p=self.initial_probabilities)]
