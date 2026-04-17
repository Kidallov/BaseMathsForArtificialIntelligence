# Walkthrough: `02-Attention / ЛР01` `GRU seq2seq + attention`

## Как пользоваться этим walkthrough
Подход тот же, что и в предыдущих self-study материалах:
1. Сначала сделать собственную попытку в notebook.
2. Потом открыть только текущую группу `TODO`.
3. Сверить формы, роли тензоров и ожидаемые промежуточные результаты.
4. Вернуться в notebook и дописать код самостоятельно.

## Что изменилось после `01-RNN / ЛР03`
Базовый `seq2seq` pipeline остаётся тем же:
- токены и словарь те же;
- `decoder_input` и `decoder_target` по-прежнему сдвинуты;
- teacher forcing остаётся.

Новое появляется в механике decoder:
- encoder теперь отдаёт всю последовательность `encoder_outputs`, а не только один итоговый вектор;
- decoder строит `query`;
- attention вычисляет `attention_scores` и `context`;
- финальный прогноз строится уже из `[decoder_outputs; context]`.

## TODO 1-3: reverse-последовательность и decoder shift
### Что нужно понять
- этот шаг практически совпадает с `01-RNN / ЛР03`;
- разница в том, что длина encoder увеличена до `ENC_LEN = 10`, чтобы bottleneck plain `seq2seq` был заметнее.

### Логика решения
- `rev = seq[::-1]`;
- `dec_in = [SOS] + rev`;
- `dec_out = rev + [EOS]`;
- итоговые формы: `(N, T_in)`, `(N, T_out)`, `(N, T_out, 1)`.

### Ожидаемые результаты
- `enc_train.shape[1] == 10`
- `dec_in_train.shape[1] == 11`
- `dec_tgt_train.shape[1:] == (11, 1)`

### Типичные ошибки
- пропустить `EOS` в `decoder_target`;
- построить `decoder_input` без `SOS`;
- забыть последнюю ось у `decoder_target`.

## TODO 4-13: encoder, decoder и attention
### Что нужно понять
- `Embedding(mask_zero=True)` нужен и encoder, и decoder;
- encoder `GRU` обязан вернуть и всю последовательность, и финальное состояние;
- decoder `GRU` тоже возвращает последовательность состояний;
- attention получает запросы от decoder и память encoder.

### Логика решения
- `enc_emb = Embedding(..., mask_zero=True)(encoder_inputs)`;
- `enc_gru = GRU(..., return_sequences=True, return_state=True)`;
- `encoder_outputs, encoder_state = enc_gru(enc_emb)`;
- `dec_emb = Embedding(..., mask_zero=True)(decoder_inputs)`;
- `decoder_outputs, _ = dec_gru(dec_emb, initial_state=encoder_state)`;
- `context, attention_scores = Attention(score_mode='dot')([decoder_outputs, encoder_outputs], return_attention_scores=True)`;
- `merged = Concatenate(axis=-1)([decoder_outputs, context])`;
- `outputs = Dense(vocab_size, activation='softmax')(merged)`.

### Ожидаемые результаты
- `model.output_shape == (None, DEC_LEN, VOCAB_SIZE)`
- в `analysis_model` доступны `encoder_outputs`, `decoder_outputs`, `context`, `attention_scores`, `outputs`
- `attention_scores.shape == (batch, T_out, T_in)` на мини-проверке.

### Типичные ошибки
- перепутать местами `decoder_outputs` и `encoder_outputs` в `Attention(...)`;
- забыть `return_sequences=True` у encoder `GRU`;
- склеить тензоры не по последней оси;
- собрать обычный `Sequential`, хотя нужен `Functional API`.

### Self-check
- Почему `Attention([decoder_outputs, encoder_outputs])` чувствителен к порядку?
  Потому что первый тензор трактуется как `query`, а второй — как `value` и `key`.

## TODO 14: обучение
### Что нужно понять
- этот шаг похож на `01-RNN / ЛР03`, но обучается уже attention-модель;
- `history.history` по-прежнему не содержит `exact_match`, только встроенные метрики Keras.

### Ожидаемые результаты
- есть `loss`, `val_loss`, `accuracy`, `val_accuracy`;
- метрики постепенно растут;
- на длинных последовательностях модель с attention обычно ведёт себя устойчивее plain baseline.

### Типичные ошибки
- передать в `fit` только один вход;
- вернуть из `train_model` не `History`, а `None`;
- перепутать порядок аргументов.

## TODO 15: `exact_match`
### Что нужно понять
- `exact_match` считается по тем же принципам, что в `01-RNN / ЛР03`;
- внимание меняет архитектуру, но не логику строгой проверки полной последовательности.

### Логика решения
- получить `preds = probs.argmax(axis=-1)`;
- построить `mask = (target != PAD_ID)`;
- сравнить `preds == target` только по значимым позициям;
- для каждого объекта проверить, совпали ли все значимые токены.

### Типичные ошибки
- случайно считать точность по токенам вместо строгого совпадения;
- забыть убрать последнюю ось у `decoder_target`;
- включить PAD-позиции в итоговое сравнение.

## Как читать heatmap после решения
1. Сначала проверить, что строки — это decoder-шаги, а столбцы — encoder-позиции.
2. Затем скрыть PAD-строки и PAD-столбцы.
3. И только после этого искать паттерн reverse-задачи.

Для этой лабораторной хороший знак:
- ранние decoder-шаги тяготеют к поздним encoder-позициям;
- позже фокус постепенно смещается к началу входа;
- общая структура похожа на антидиагональ, а не на хаотический шум.

## Что должно получиться в конце
После `02-Attention / ЛР01` студент должен уметь объяснить:
- зачем нужны `encoder_outputs`, если уже есть `encoder_state`;
- что в этой модели играет роли `query`, `key`, `value`;
- почему `attention_scores` имеют форму `(N, T_out, T_in)`;
- как читать heatmap и отличать разумный паттерн от шумного;
- почему эта лабораторная логично ведёт к следующей теме про `Transformer`.

## Навигация по курсу
### Где вы сейчас
`02-Attention / ЛР01` — локальная ЛР01 блока `Attention`, Шаг 4 общего курса.

### Что было до этого
`01-RNN / ЛР03` — plain `GRU seq2seq` без внимания.

### Что дальше
Следующая тема курса — `Transformer`. В ней внимание уже не надстройка над RNN, а базовый механизм всей модели.
