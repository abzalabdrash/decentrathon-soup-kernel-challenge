# Для тиммейта: подключение и роль

## Что это за задача

Soup Kernel Challenge, трек 01 Decentrathon 6.0. Организаторы дали фиксированный прогон обучения (`soup train`: Qwen2.5-1.5B в 4 битах, LoRA r=16, 256 казахских инструкций, 64 шага) и харнесс, который на Kaggle T4 запускает его дважды: базовый Soup и Soup с нашим пакетом `solution/` (функция `apply()` патчит transformers, peft, bitsandbytes, trl). Считают два отношения: время базы к нашему и пик памяти базы к нашему. Обучение должно остаться тем же (кривая loss в пределах 3%, eval loss не хуже чем на 0,5%, те же метрики и rewind), иначе ноль. Базу ниоткуда скачивать не надо: харнесс сам её прогоняет в той же сессии перед нашим решением. Подробности: `README.md`, правила и 10 проверок там же, план работ в `docs/TASKS.md`.

## Установка (Linux или WSL2, карта NVIDIA)

```bash
git clone https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers.git soup && cd soup
python3 -m venv .venv && source .venv/bin/activate
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
sudo apt-get install -y build-essential   # gcc нужен Triton для сборки лаунчера
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Версии torch и triton подтянутся зависимостями (на Kaggle torch 2.11, triton 3.6.0). Если карта Ampere или новее, Soup будет учить в bf16: это другой путь, числа времени и памяти с T4 не сравнимы, но корректность и падения ловятся.

## Что делать на ноутбуке с 8 ГБ

База (14 ГБ) не влезет, наше решение по оценке влезет. После каждого обновления main:

```bash
git pull
python harness/run_task.py --only solution --out runs_try
```

Смотреть `runs_try/solution_1/train.log` и `runs_try/result.json`: обучение дошло до конца, 64 шага, `rewind.jsonl` записан, в логе нет исключений. Любое падение или странность сразу в чат команды с хвостом лога. Это ловит ошибки интеграции до того, как мы потратим 12 минут на Kaggle.

## Промпт для своей сессии Claude (скопировать целиком)

Ты работаешь в репозитории команды хакатона Decentrathon 6.0, трек Soup Kernel Challenge. Прочитай README.md (правила, 10 проверок), docs/TASKS.md (задачи и границы), solution/fused_ce.py и solution/unpad.py (стиль патчей и защит). Твоя роль: взять задачу `<КОД ЗАДАЧИ ИЗ docs/TASKS.md>` и довести её до состояния «готово» из таблицы: своя ветка `task/<код>`, только свои файлы в solution/, тесты рядом, микробенчмарк против eager PyTorch на формах задачи, проверка `python harness/check_rules.py` на похожесть с примером. Ограничения: ядра только свои `@triton.jit` в solution/, без tl.dot для больших матмулов (на T4 Triton без тензорных ядер), без готовых библиотек ядер (unsloth, liger, xformers, flash-attn, torchao и список в harness/common.py), без torch.compile, без изменения задачи обучения; каждый патч должен откатываться на оригинал, если на вход пришло не то (тесты Soup вызывают патченый код с фальшивыми модулями). Локальная карта только для корректности и грубого времени, итоговые числа с Kaggle T4. Работай до готовности, после каждого значимого шага пиши короткий отчёт: файлы, результаты тестов, таблица времени, что не получилось. Не трогай solution/__init__.py, harness/, tests/, task/.
