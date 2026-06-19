import math
import numpy as np
import matplotlib.pyplot as plt


def power_scale(
    count: int,
    min_val: int = 3,
    max_val: int = 15,
    max_count: int = 35000,
    power: float = 0.5,
) -> int:
    """ Экспоненциальная функция масштабирования.
        Используется для получения общего кол-ва фраз в LengthSonifier"""

    if count <= 0:
        return min_val
    progress = (count / max_count) ** power
    return int(round(min(max_val, min_val + (max_val - min_val) * progress)))


def log_scale(
    count: int,
    min_val: int = 3,
    max_val: int = 15,
    max_count: int = 28000,
) -> int:
    """ Логарифмическая функция масштабирования.
        Используется для получения длины фразы в LengthSonifier и уровня громкости (velocity) в VelocitySonifier"""
    if count <= 0:
        return min_val
    progress = math.log(1 + count) / math.log(1 + max_count)
    return int(round(min(max_val, min_val + (max_val - min_val) * progress)))



x = np.arange(0, 35001, 100)

power_y = [power_scale(v) for v in x]
log_y = [log_scale(v) for v in x]
linear_y = [int(round(min(15, 3 + (15 - 3) * v / 35000))) for v in x]

plt.figure(figsize=(10, 5))
plt.plot(x, power_y, label="power_scale (power=0.5)")
plt.plot(x, log_y, label="log_scale")
plt.plot(x, linear_y, label="linear (для сравнения)", linestyle="--")
plt.xlabel("Количество отзывов")
plt.ylabel("Количество фраз")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("scaling_comparison.png", dpi=150)