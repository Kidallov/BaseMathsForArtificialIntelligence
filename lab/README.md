# Лабораторный практикум по Attention

## Назначение
Практикум содержит одно задание на синтетических данных:
1. `GRU` encoder-decoder с `Luong attention` для `seq2seq`.

Цель практикума — показать, как `attention` снимает fixed-context bottleneck обычного `seq2seq`, научить читать тензоры `query/key/value`-логики и интерпретировать `attention_scores` через тепловую карту.

## Структура
- `01_gru_seq2seq_attention_reverse_toy.ipynb` — задание с `TODO`.
- `solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb` — полное решение.

Основной ноутбук и решение синхронизированы 1:1: одинаковая структура разделов и одинаковые контрольные точки.

## Карта курса
Этот блок не открывает “новую ЛР03”, а продолжает уже пройденную линию:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`

То есть текущая лабораторная — это первая локальная ЛР блока `Attention`, но четвёртый шаг общего курса.

## Beginner Route
Рекомендуемый маршрут для новичка:
1. Пройти общий shared-entry: [../../00-Foundations/README.md](../../00-Foundations/README.md).
2. Повторить токены, `PAD/SOS/EOS`, masking и decoder shift по shared-guide:
   [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md)
3. Повторить базовую интуицию heatmap и `query/key/value` по shared-guide:
   [../../00-Foundations/guides/03_attention_heatmaps.md](../../00-Foundations/guides/03_attention_heatmaps.md)
4. При необходимости свериться с RNN-specific guide из `01-RNN / ЛР03`:
   [../../01-RNN/lab/guides/03_gru_seq2seq_tokens_beginner.md](../../01-RNN/lab/guides/03_gru_seq2seq_tokens_beginner.md)
5. Прочитать [guides/00_attention_prerequisites.md](./guides/00_attention_prerequisites.md).
6. Прочитать [guides/01_gru_seq2seq_attention_beginner.md](./guides/01_gru_seq2seq_attention_beginner.md).
7. При необходимости пройти [guides/02_attention_walkthrough.md](./guides/02_attention_walkthrough.md) и [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md).
8. Только после этого открывать Шаг 4 = `02-Attention / ЛР01`.

## Self-study Route
Если `02-Attention / ЛР01` нужно пройти полностью без преподавателя, используйте такой маршрут:
1. [../../00-Foundations/README.md](../../00-Foundations/README.md)
2. [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md)
3. [../../00-Foundations/guides/03_attention_heatmaps.md](../../00-Foundations/guides/03_attention_heatmaps.md)
4. [../../01-RNN/lab/guides/03_gru_seq2seq_tokens_beginner.md](../../01-RNN/lab/guides/03_gru_seq2seq_tokens_beginner.md)
5. [guides/00_attention_prerequisites.md](./guides/00_attention_prerequisites.md)
6. [guides/01_gru_seq2seq_attention_beginner.md](./guides/01_gru_seq2seq_attention_beginner.md)
7. [guides/02_attention_walkthrough.md](./guides/02_attention_walkthrough.md)
8. [01_gru_seq2seq_attention_reverse_toy.ipynb](./01_gru_seq2seq_attention_reverse_toy.ipynb)
9. [solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb](./solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb)
10. [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md)

## Companion Guides
- [../../00-Foundations/README.md](../../00-Foundations/README.md) — общий entry-point перед sequence-лабораторными и shared warm-up materials.
- [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md) — общий seq2seq-вход: словарь, `PAD/SOS/EOS`, masking и teacher forcing.
- [../../00-Foundations/guides/03_attention_heatmaps.md](../../00-Foundations/guides/03_attention_heatmaps.md) — короткий shared-guide по `query/key/value`, score-matrix и чтению heatmap.
- [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md) — единый self-study playbook для sequence и attention notebook'ов.
- [guides/00_attention_prerequisites.md](./guides/00_attention_prerequisites.md) — bottleneck plain `seq2seq`, `query/key/value`, `context`, `attention_scores`, чтение heatmap.
- [guides/01_gru_seq2seq_attention_beginner.md](./guides/01_gru_seq2seq_attention_beginner.md) — лабораторно-специфический разбор reverse-задачи с attention.
- [guides/02_attention_walkthrough.md](./guides/02_attention_walkthrough.md) — пошаговый разбор всех `TODO` `02-Attention / ЛР01`.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/02-Attention/lab/requirements.txt
python3 -m ipykernel install --user --name attention-lab --display-name "Python (.venv) Attention Lab"
.venv/bin/jupyter notebook
```

### Runtime Варианты
TensorFlow notebook'и этого блока теперь можно запускать:
- локально через `auto`, `local-cpu`, `local-gpu`;
- в `Google Colab` через `colab-cpu` или `colab-gpu`;
- в `Kaggle` через `kaggle-cpu` или `kaggle-gpu`.

Если нужен понятный выбор между локальным запуском, `Colab`, `Kaggle`, `CPU` и `GPU`, используйте общий guide:
[../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)

Если вы хотите именно локальный GPU и не уверены, как думать про версии `TensorFlow` / `CUDA`, используйте:
[../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)

## Порядок прохождения
1. Шаг 4 = `02-Attention / ЛР01` -> `01_gru_seq2seq_attention_reverse_toy.ipynb`

## Что читать перед лабораторной
| Лабораторная | Что открыть до ноутбука |
|---|---|
| `02-Attention / ЛР01` `GRU seq2seq + attention` | `../../00-Foundations/README.md` + `../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md` + `../../00-Foundations/guides/03_attention_heatmaps.md` + guide из `01-RNN / ЛР03` про токены + `00_attention_prerequisites.md` + `01_gru_seq2seq_attention_beginner.md` + `02_attention_walkthrough.md` |

## Как читать формы тензоров
Базовый шаблон:
- `N` — число объектов в наборе;
- `T_in` — длина входной последовательности encoder;
- `T_out` — длина выходной последовательности decoder;
- `H` — размер скрытого пространства;
- `V` — размер словаря.

Ключевой принцип:
- `encoder_outputs` хранит представления всех входных позиций, а не только одно финальное состояние;
- `attention_scores` имеет форму `(batch, T_out, T_in)` и показывает, куда decoder смотрит на каждом шаге.

Проверка на каждом шаге:
1. До обучения: сравнить фактические формы `encoder_input`, `decoder_input`, `decoder_target`.
2. После сборки модели: проверить форму выхода `model.output_shape`.
3. После `predict`: проверить формы `context`, `attention_scores`, `probs`.

## Методический маршрут
Рекомендуемый цикл работы в ноутбуке:
1. `Что нужно знать до старта` и `Интуиция задачи без формул`.
2. `Контракт данных`, `Таблица форм тензоров`, `Шпаргалка по обозначениям и формам`.
3. `Контракт модели`, `Мини-теория`, `Ручной разбор одного примера`.
4. `Генерация данных` -> `Мини-проверка данных`.
5. `Модель` -> `Мини-проверка модели` -> `Трассировка одного примера`.
6. `Как идет обучение внутри эпохи` -> `Обучение` -> `Мини-проверка обучения`.
7. `Оценка и диагностика` -> `Как читать графики и метрики`.
8. `Карта внимания одного примера`.
9. `Если застряли: порядок диагностики` -> `Чек-лист перед сдачей` -> `Вопросы для самопроверки`.

## Схема валидации
В работе используется тот же подход, что и в блоке `RNN`:
- сначала формируется отдельный `test` через `train_test_split`;
- затем обучение выполняется на `train` с `validation_split` внутри `model.fit`.

Это разделяет:
- настройку обучения (`train`/`validation`),
- финальную независимую оценку (`test`).

## API-конвенция
Лабораторная полностью использует `Functional API`, потому что attention требует:
- два входа модели;
- промежуточные тензоры `encoder_outputs`, `decoder_outputs`, `context`;
- отдельный доступ к `attention_scores` для анализа.

## Связь с теорией
- Раздел 1 в `theory.md` -> зачем plain `seq2seq` нужен attention.
- Раздел 2 -> интуиция `query`, `key`, `value`.
- Раздел 3 -> формулы Luong attention и формы тензоров.
- Раздел 4 -> отличие `cross-attention` от `self-attention`.
- Раздел 5 -> почему следующий шаг после этой ЛР — облегченный `Transformer`.

`theory.md` остаётся канонической теорией, а guides дают новичку мягкий вход, ручные примеры и пошаговую интерпретацию attention.

## Дополнительные материалы
- [../theory/theory.md](../theory/theory.md) — компактная теория по attention и мостик к `Transformer`.
- [../../00-Foundations/README.md](../../00-Foundations/README.md) — общий shared-entry, warm-up examples и optional real-data showcases.
- TensorFlow tutorial: NMT with attention — <https://www.tensorflow.org/tutorials/text/nmt_with_attention>
- Keras layer API: `Attention` — <https://keras.io/api/layers/attention_layers/attention/>
- TensorFlow tutorial: Transformer — <https://www.tensorflow.org/text/tutorials/transformer>

## FAQ
### Я не понимаю, где заканчивается `seq2seq` и начинается attention
Сначала вернитесь к guide из `01-RNN / ЛР03` про `seq2seq`-токены и сдвиг decoder. Потом прочитайте `00_attention_prerequisites.md`: attention — это надстройка над уже понятным `seq2seq`.

### Нужно ли заново изучать токенизацию в этой лабораторной?
Полный вводный блок про токены уже должен быть освоен в `01-RNN / ЛР03`. В attention-лаборатории он не дублируется полностью, а только напоминается.

### Как понять, что attention действительно работает?
Нужно смотреть не только на `token_accuracy` и `exact_match`, но и на `attention_scores` / heatmap. Для reverse-задачи хороший знак — фокус примерно по антидиагонали.

### Что читать после этой ЛР?
Следующий естественный шаг — блок [../../03-Transformer/lab/README.md](../../03-Transformer/lab/README.md), начиная с `03-Transformer / ЛР01`.

## Типичные проблемы
### Attention-модель
- Симптом: `attention_scores` имеет неожиданную форму.
  Причина: перепутаны местами `query` и `value`.
  Исправление: вызывать `Attention([decoder_outputs, encoder_outputs])`.
- Симптом: attention не может смотреть по всем входным позициям.
  Причина: encoder возвращает только финальное состояние.
  Исправление: включить `return_sequences=True` у `enc_gru`.

### Метрики и интерпретация
- Симптом: `token_accuracy` растет, а `exact_match` заметно отстает.
  Причина: ошибки остаются хотя бы в одном шаге многих последовательностей.
  Исправление: проверить сборку модели, число эпох и корректность teacher forcing.
- Симптом: heatmap выглядит шумной и плохо читается.
  Причина: PAD-позиции не скрыты при визуализации.
  Исправление: показывать только значимые строки и столбцы.
- Симптом: неясно, что такое `query`, `key`, `value`.
  Причина: attention читается как набор новых слов без интуитивного образа.
  Исправление: начать с `00_attention_prerequisites.md`, а не сразу с кода.

### Сравнение с plain seq2seq
- Симптом: разница с baseline почти не видна.
  Причина: слишком короткие последовательности или слишком легкая задача.
  Исправление: использовать `ENC_LEN=10` и переменную длину `4..10`, как в ноутбуке.
