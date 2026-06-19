import numpy as np
from sonification.__main__ import sonify
from sonification.__main__ import show
import json

output_dir = "experiments"

sonifier = Sonifier()
log = []


def generate_lengths(avg_len, count):
     return sorted(np.random.exponential(scale=avg_len, size=count).astype(int).tolist())

sentiment_lengths = [
    {
         "positive": generate_lengths(50, 20000),
          "neutral": generate_lengths(20, 3000),
          "negative": generate_lengths(100, 7000)
     },
    {
         "positive": generate_lengths(50, 200),
         "neutral": generate_lengths(30, 300),
         "negative": generate_lengths(10, 250)
    },
]


for lengths in sentiment_lengths:
     stats = BrandStats(
          name='',
          sentiment_lengths=lengths,
     )
     total = stats.total_reviews
     prob = stats.sentiment_probabilities
     name = f"vel_{total}_{max(prob, key=prob.get)}_{len(lengths['positive'])}"
     stats.name = name
     melody = sonifier.sonify_by_velocity(stats)
     score = Sonifier.to_score(melody, title=name)
     score.write("midi", str(f"{output_dir}/{name}.mid"))

     log.append({
          "name": name,
          "total_reviews": total,
          "probabilities": prob,
          "lengths_count": {s: len(l) for s, l in lengths.items()},
          "melody_notes": len(melody),
     })

with open(f"{output_dir}/log.json", "w") as f:
    json.dump(log, f)
