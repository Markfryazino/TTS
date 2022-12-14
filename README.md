# Deep Learning in Audio. Домашнее задание по FastSpeech2.

## Инструкция по запуску

1. Клонируем репозиторий.
    ```
    git clone https://github.com/Markfryazino/TTS.git
    ```

1. Поднимаем контейнер. `WANDB_API_KEY` можно не указывать. Все операции далее, разумеется происходят внутри контейнера.
    ```
    cd TTS
    docker build --network host -t tts .
    docker run -it --network host -e WANDB_API_KEY=YOUR_API_KEY -v ./TTS:/repos/tts_project tts
    ```

1. Скачиваем сторонние данные.
    ```
    bash scripts/prepare_data.sh
    ```

1. Скачиваем чекпоинт модели и посчитанные значения pitch/energy для LJSpeech.
    ```
    bash scripts/download_artifacts.sh
    ```
    Чтобы вопроизвести вычисление pitch/energy, если вдруг хочется, можно запустить для этого [скрипт](./compute_pitch_energy.py).
    ```
    python3 compute_pitch_energy.py
    ```
    Но он работает примерно полчаса.

1. Теперь можно запустить модель. Для этого есть скрипт [test.py](./test.py). У него там много всяких параметров, но достаточно сделать вот так:
    ```
    python3 test.py --model data/download/data/cool_model/checkpoint_140000.pth.tar --texts test_texts.json
    ```
    Этот запуск создаст папку `test_wavs` и положит туда аудиозаписи, сгенерированные по текстам из [test_texts.json](./test_texts.json) с разными параметрами duration/pitch/energy. Все эти аудиозаписи можно найти и в [WandB](https://wandb.ai/broccoliman/TTS/runs/zhjx42xu).

1. Чтобы вопроизвести обучение, можно запустить [train.py](./train.py).
    ```
    python3 train.py
    ```

## Немного про содержание репозитория

Пакет с моделью и всеми содержательными классами лежит в папке [src](./src).

В основном код, конечно, скопирован с семинара по FastSpeech. Основной мой вклад в модель -- это файлы [FFT.py](./src/model/FFT.py) (я поменял FFT блок на более эффективный) и [VarianceAdaptor.py](./src/model/VarianceAdaptor.py) (собственно Variance Adaptor из статьи о FastSpeech2).

В папке [src/xcmyz_utils](./src/xcmyz_utils/) лежат функции из китайского репозитория по FastSpeech, которые я счёл возможным использовать, так как это делалось и на семинаре. Оттуда же файл [glow.py](./glow.py): его пришлось положить в корневую директорию, потому что иначе не получается загрузить WaveGlow.