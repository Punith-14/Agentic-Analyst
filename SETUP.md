# Setup — sub-problem B

Windows, PowerShell. Do these in order.

---

## 0. Move the project out of OneDrive first

**This is not optional.** OneDrive syncs files while SQLite is writing to them,
which causes `disk I/O error` and `attempt to write a readonly database`. It
will break your overnight trajectory runs.

```powershell
# move the code and data somewhere unsynced
mkdir C:\dev
move "C:\Users\punit\OneDrive\Desktop\AIML_BootCamp\project23" C:\dev\
cd C:\dev\project23
```

Keep documents and research notes in OneDrive. Keep code and databases out.

Also delete these leftovers if they are still there:

```powershell
del data\db\_t.db, data\db\_t.db-journal, data\db\formula_1.db-journal
```

A stale `-journal` file makes SQLite refuse read-only connections.

---

## 1. Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

You should see `(.venv)` at the start of your prompt. It must be there every
time you work on the project.

---

## 2. Core dependencies

```powershell
pip install -r requirements.txt
```

That is everything needed for data prep and notebook 01.

---

## 3. Register the kernel and open the notebook

```powershell
python -m ipykernel install --user --name project23 --display-name "Project 23"
jupyter lab
```

In JupyterLab open `notebooks/01_dataset_exploration.ipynb` and pick the
**Project 23** kernel from the top-right, then Run All.

---

## 4. Rebuild the data (only if you need to)

The databases and task suite are already built. To redo them:

```powershell
python data_prep\prepare_spider.py --spider-dir Data\spider_data --list

python data_prep\prepare_spider.py --spider-dir Data\spider_data `
    --dbs formula_1,college_2 --holdout chinook_1 --n 40 --n-holdout 20
```

---

## 5. ML stack — week 2, not now

**PyTorch first, separately, with CUDA.** The default pip install gives you the
CPU build and fine-tuning will crawl.

```powershell
nvidia-smi                       # note your CUDA version
# then take the command from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.cuda.is_available())"
```

That must print `True`. If it prints `False`, stop and fix it — everything
downstream depends on the GPU working.

Then:

```powershell
pip install -r requirements-ml.txt
```

### The 7B model

Simplest route is Ollama (desktop app, no compiler needed):

```powershell
# install from https://ollama.com then:
ollama pull qwen2.5-coder:7b
ollama run qwen2.5-coder:7b "SELECT 1"
```

To use 8-bit KV cache (halves cache memory, <0.1% quality loss):

```powershell
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
$env:OLLAMA_FLASH_ATTENTION="1"
# restart Ollama after setting these
```

---

## 6. Folder layout

```
project23/
  contracts.py                    shared schemas — DO NOT EDIT ALONE
  requirements.txt                core
  requirements-ml.txt             week 2
  data_prep/prepare_spider.py     builds databases + task suite
  notebooks/
    01_dataset_exploration.ipynb  done
    02_trajectory_features.ipynb  week 2, needs real runs
  data/
    db/*.db                       three SQLite databases
    tasks/task_suite.json         60 questions with gold SQL
    tasks/schemas.json            schema summary per database
    trajectories/*.jsonl          agent runs — NEVER DELETE
  agent/                          week 1: loop, parser, logger
  tests/
  .env                            gitignored
```

---

## 7. Quick check that everything works

```powershell
python -c "import contracts; print('contracts ok')"
python -c "import json; t=json.load(open('data/tasks/task_suite.json')); print(len(t),'tasks')"
python -c "import sqlite3,pandas as pd; c=sqlite3.connect('file:data/db/chinook_1.db?mode=ro',uri=True); print(pd.read_sql_query('SELECT COUNT(*) n FROM albums',c))"
```

All three should run without error.
