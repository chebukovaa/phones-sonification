# Сонификация данных о телефонах с Amazon

Генерация музыкальных мелодий на основе анализа тональности отзывов на мобильные телефоны с Amazon. Программа преобразует статистику отзывов каждого бренда в короткий музыкальный фрагмент — позитивные отзывы звучат мажорно, негативные — напряжённо, а количество и длина отзывов кодируются через длительность или громкость мелодии.

## Требования

- Python 3.10+
- FluidSynth (`sudo apt install fluidsynth`)
- SoundFont: `/usr/share/sounds/sf2/FluidR3_GM.sf2` (устанавливается вместе с FluidSynth на большинстве дистрибутивов)

## Установка

```bash
git clone https://github.com/username/phones-sonification.git
cd phones-sonification
pip install -r requirements.txt
```

## Данные

Датасет [Amazon Unlocked Mobile Phones](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-unlocked-phone) доступен на Kaggle.

### Вариант 1 — через Kaggle CLI

1. Получите API-токен: зайдите на kaggle.com → Account → Create New Token. Файл `kaggle.json` с токеном API сохранится в `~/.kaggle/kaggle.json`.

2. Установите Kaggle CLI:
   ```bash
   pip install kaggle
   ```

3. Скачайте и распакуйте датасет (выполнить из корня проекта, папка "/phones-sonification"):
   ```bash
   kaggle datasets download -d PromptCloudHQ/amazon-unlocked-phone --unzip -p data/
   ```

### Вариант 2 — вручную

1. Перейдите на страницу датасета: https://www.kaggle.com/datasets/PromptCloudHQ/amazon-unlocked-phone
2. Нажмите Download.
3. Распакуйте архив и поместите файл `Amazon_Unlocked_Mobile.csv` в директорию `data/`:
   ```
   .../phones-sonification/data/Amazon_Unlocked_Mobile.csv
   ```

## Использование

Пайплайн состоит из двух этапов.

**Шаг 1 — предобработка и классификация тональности** (выполняется однократно, занимает значительное время):

```bash
python -m sonification prepare
```

Результат сохраняется в `data/phones_data.parquet`.

**Шаг 2 — генерация мелодий:**

```bash
python -m sonification sonify
```

WAV-файлы сохраняются в `output/sonified_brands/`. Для каждого бренда генерируются два файла: `<brand>_LengthSonifier.wav` и `<brand>_VelocitySonifier.wav`.

### Опции командной строки

```
python -m sonification prepare
  --raw-csv-path PATH   путь к исходному CSV (по умолчанию: data/Amazon_Unlocked_Mobile.csv)
  --parquet-path PATH   путь для сохранения Parquet (по умолчанию: data/phones_data.parquet)

python -m sonification sonify
  --parquet-path PATH   путь к Parquet-файлу (по умолчанию: data/phones_data.parquet)
  --output-dir PATH     директория для WAV-файлов (по умолчанию: output/sonified_brands)
```

## Веб-интерфейс

```bash
streamlit run server.py
```

Путь к директории с результатами можно переопределить через переменную окружения:

```bash
SONIFICATION_OUTPUT_DIR=output/sonified_brands streamlit run server.py
```

## Структура проекта

```
sonification/             # пакет, выполняющий сонификацию
├── __main__.py           # точка входа, команды prepare и sonify
├── data_loader.py        # загрузка и предобработка CSV
├── sentiment_classifier.py  # классификация тональности (RoBERTa)
├── brand_analyzer.py     # группировка отзывов по брендам
├── brandstats.py         # датакласс BrandStats
├── sonifier.py           # LengthSonifier и VelocitySonifier
├── markov_models.py      # seed-мелодии и обученные модели
├── markov_generator.py   # генератор мелодий на цепях Маркова
├── scaling.py            # функции масштабирования
└── sentiment.py          # перечисление Sentiment
server.py                 # веб-интерфейс Streamlit
```

## Алгоритмы сонификации

В проекте реализовано 2 алгоритма:

**LengthSonifier** — кодирует количество отзывов через длину мелодии, длину отзывов — через длину отдельных фраз. Фразы разных тональностей перемешиваются случайно.

**VelocitySonifier** — фиксированное число нот (20), количество отзывов кодируется через громкость (velocity). Фразы следуют блоками по тональности.

В обоих алгоритмах характер мелодии (мажорный, минорный или хроматический) определяется долей позитивных, нейтральных и негативных отзывов через предобученные модели цепей Маркова.
