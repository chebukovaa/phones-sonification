from __future__ import annotations

from dataclasses import dataclass

from random import shuffle

import numpy as np
from music21 import metadata, note, stream

from .sentiment import Sentiment
from .markov_generator import MarkovChainMelodyGenerator
from .brandstats import BrandStats
from .markov_models import get_trained_models
from .scaling import power_scale, log_scale


@dataclass
class PhraseParams:
    sentiment: Sentiment
    n_notes: int
    velocity: int = 80


class BaseSonifier:
    def __init__(self, models=None):
        self.models = models or get_trained_models()

    def sonify(self, stats: BrandStats) -> list[tuple]:
        plan = self._build_plan(stats)
        melody = []
        last_note = None
        for phrase_params in plan:
            phrase = self.models[phrase_params.sentiment].generate_phrase(phrase_params.n_notes, last_note)
            last_note = phrase[-1]
            for pitch, duration in phrase:
                melody.append((pitch, duration, phrase_params.velocity))
            melody.append(("rest", 1.0))
        return melody

    def _build_plan(self, stats):
        raise NotImplementedError

    @staticmethod
    def to_score(melody: list[tuple[str, float]] | list[tuple[str, float, int]],
                 title: str = "Sonification") -> stream.Score:
        score = stream.Score()  # создаём партитуру
        score.metadata = metadata.Metadata(title=title)  # подписываем название
        part = stream.Part()  # создаём партию

        for item in melody:  # распаковываем кортеж с тоном и длительностью
            if item[0] == "rest":
                part.append(note.Rest(quarterLength=item[1]))  # добавляем паузу
            else:
                n = note.Note(item[0], quarterLength=item[1])  # превращаем в ноту
                n.volume.velocity = item[2]
                part.append(n)
        score.append(part)  # добавляем партию в партитуру
        return score


class LengthSonifier(BaseSonifier):
    def _build_plan(self, stats: BrandStats) -> list[PhraseParams]:
        total_phrases = power_scale(stats.total_reviews)  # определяем общее количество фраз
        plan = []

        for sentiment, prob in stats.sentiment_probabilities.items():  # для каждого сантимента
            n_phrases = int(round(total_phrases * prob))  # считаем количество фраз одного сантимента
            if n_phrases == 0 and prob != 0:
                n_phrases = 1

            review_lengths = stats.sentiment_lengths.get(sentiment, [])  # ищем длины отзывов текущего сантимента по ключу,
            # если отзывов такого сантимента нет,
            # возвращаем пустой список
            if not review_lengths or n_phrases == 0:  # если фрагментов нет - пропускаем
                continue

            if n_phrases >= len(review_lengths):  # если вдруг фраз больше, чем отзывов (хотя такого не должно быть)
                notes_per_phrase = [log_scale(review_length, 3, 10, 10000) for review_length in review_lengths]  # выбираем все и скейлим
            else:
                indices = np.linspace(0, len(review_lengths) - 1, n_phrases).astype(int)  # иначе выбираем индексы линейно
                notes_per_phrase = [log_scale(review_lengths[i], 3, 10, 10000) for i in indices]  # скейлим

            for n_notes in notes_per_phrase:  # сохраняем сантимент фразы и её длину, перемешиваем
                plan.append(PhraseParams(sentiment=sentiment, n_notes=n_notes))

        shuffle(plan)  # перемешиваем
        return plan


class VelocitySonifier(BaseSonifier):
    def _build_plan(self, stats: BrandStats) -> list[PhraseParams]:
        total_notes = 20  # задаём общее количество нот
        velocity = log_scale(stats.total_reviews, 40, 120, 35000)  # определяем громкость по количеству отзывов
        plan = []

        for sentiment, prob in stats.sentiment_probabilities.items():  # для каждого сантимента
            n_notes = int(round(total_notes * prob))  # считаем количество нот
            if prob != 0 and n_notes == 0:
                n_notes = 3
            plan.append(PhraseParams(sentiment=sentiment, n_notes=n_notes, velocity=velocity))

        # не перемешиваем
        return plan
