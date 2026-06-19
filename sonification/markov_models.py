from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sonification.markov_generator import MarkovChainMelodyGenerator
from sonification.sentiment import Sentiment


STATES = [
    (p, d) for d in (1, 2, 0.5)
    for p in ("G#4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "G#5", "A5", "B5", "C6")
]

SEED_MELODIES: dict[Sentiment, list[tuple[str, float]]] = {
    Sentiment.POSITIVE: [
        ("G5", 1), ("E5", 0.5), ("C5", 0.5), ("G5", 1), ("E5", 0.5), ("C5", 0.5),
        ("D5", 1), ("E5", 0.5), ("F5", 0.5), ("G5", 0.5), ("F5", 0.5),
        ("E5", 0.5), ("D5", 0.5), ("C5", 2),
    ],
    Sentiment.NEUTRAL: [
        ("E5", 0.5), ("C6", 1), ("B5", 0.5), ("A5", 0.5), ("B5", 0.5),
        ("C6", 0.5), ("B5", 1), ("A5", 0.5), ("G#5", 0.5), ("A5", 0.5),
        ("B5", 0.5), ("A5", 1), ("F5", 0.5), ("E5", 0.5), ("D5", 0.5), ("E5", 1),
    ],
    Sentiment.NEGATIVE: [
        ("D5", 1), ("G#5", 1), ("A5", 1), ("C5", 0.5), ("B5", 0.5), ("G#5", 2),
        ("A5", 1), ("E5", 0.5), ("F5", 0.5), ("B4", 1), ("C5", 1), ("A4", 1),
        ("G#4", 0.5), ("B4", 0.5), ("D5", 0.5), ("F5", 0.5), ("E5", 2),
        ("A5", 1), ("G#5", 2), ("E5", 1), ("F5", 0.5), ("D5", 0.5),
        ("B4", 0.5), ("G#4", 0.5), ("A4", 1),
    ],
}


def get_trained_models() -> dict[Sentiment, MarkovChainMelodyGenerator]:
    models = {}
    for sentiment, notes in SEED_MELODIES.items():
        models[sentiment] = MarkovChainMelodyGenerator(STATES).train(notes)
    return models


if __name__ == "__main__":
    from music21 import stream, note as m21note
    for sentiment, notes in SEED_MELODIES.items():
        s = stream.Part()
        for pitch, duration in notes:
            s.append(m21note.Note(pitch, quarterLength=duration))
        s.write('lily.png', fp=f'seed_{sentiment.value}.png')