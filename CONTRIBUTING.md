# How we work on this repo

**Team Mind Matrix · Project 23 — Agentic Analyst**

Repo: <https://github.com/Punith-14/Agentic-Analyst>

Everyone on the team follows this document. It exists so that four people can
work on the same codebase at the same time without overwriting each other, and
so that when something breaks we can tell who changed what and undo it.

Read it once fully. After that, the section you will use every day is
[Your daily loop](#your-daily-loop).

---

## Who owns what

| Layer | Owner | Scope | Branch prefix |
|---|---|---|---|
| A | Dhrub | Tool library, sandbox, Pydantic validation | `dhrub/` |
| B | Punith | ReAct loop, trajectory logger, learned critic | `punith/` |
| C | Krishna | Memory — short-term, episodic, semantic | `krishna/` |
| D | Harish | LangGraph orchestration, React UI | `harish/` |

**You only edit files inside your own layer.** If you need something changed in
someone else's folder, ask them — don't edit it yourself. The one exception is
`contracts.py`, which has its own rule below.

---

## The six rules

1. **Never commit directly to `main`.** All work goes through a branch and a
   pull request. `main` is protected, so git will refuse anyway.
2. **Punith merges to `main`.** You open the PR; he presses the button. See
   [Who merges](#who-merges-and-when) for the full rule, including what happens
   if he doesn't respond.
3. **Never use `git push --force`.** It rewrites history and can delete a
   teammate's work permanently. If you think you need it, ask first.
4. **Pull before you start work, every single time.** Most merge conflicts are
   caused by skipping this.
5. **Never commit data, databases, secrets, or your `.venv`.** The `.gitignore`
   handles this — don't override it with `git add -f`.
6. **`contracts.py` is shared.** Changing it changes the code of all four
   people. Tell the group chat *before* you open the PR, not after.

---

## One-time setup

Do this once, on each machine you work on.

### 1. Tell git who you are

Your name shows up on every commit. Use your real name so contributions are
attributable at evaluation time.

```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

Use the **same email as your GitHub account**, otherwise your commits won't be
linked to your profile and it will look like you did nothing.

### 2. Clone the repo

Pick a folder with a short path and **not** inside OneDrive, Google Drive or
Dropbox. Cloud sync corrupts SQLite databases and git index files. We lost time
to this already.

Good: `C:\dev\` · Bad: `C:\Users\you\OneDrive\Desktop\`

```bash
cd C:\dev
git clone https://github.com/Punith-14/Agentic-Analyst.git
cd Agentic-Analyst
```

### 3. Set up Python

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
pip install -r requirements-ml.txt    # only if you train models
```

### 4. Build the data you need

The large data files are **not** in the repo — they are 1.8 GB and rebuildable.

```bash
python data_prep/prepare_spider.py
```

### 5. Check it works

```bash
pytest -q
```

If the tests pass, you're set up correctly. If they don't, say so in the group
chat before you start changing things — a broken setup and a broken `main` look
identical from the inside.

---

## Branch naming

**Format:**

```
<your-name>/<short-description>
```

- `<your-name>` is your first name, lowercase: `dhrub`, `punith`, `krishna`,
  `harish`
- `<short-description>` is two to four words, **lowercase**, separated by
  hyphens
- No spaces, no capitals, no underscores

**Good:**

```
dhrub/sandbox-timeout
dhrub/pydantic-tool-validation
punith/critic-early-stopping
punith/fix-parser-repair
krishna/episodic-memory-store
harish/langgraph-router
harish/results-table-ui
```

**Bad:**

```
main                       ← never work here
krishna                    ← which task? you'll have twenty of these
new-branch                 ← meaningless in a month
Harish/Fix_The_UI_Bug      ← capitals and underscores
punith/fix                 ← fix what?
```

**Why your name in front:** GitHub sorts branches alphabetically, so all of your
work groups together and anyone can see who owns a branch without opening it.

**Why a description after it:** git already records *who* wrote every commit, in
the author field. What git does **not** record is what the work was. A branch
called just `krishna` spends its name on information we already have.

**One branch = one piece of work.** When that work is merged, the branch is
finished — start a new one for the next piece. Do not keep a single
`krishna/work` branch alive for two weeks. A branch that lives that long
produces a pull request nobody can review, and it drifts further from `main`
every day you leave it.

---

## Your daily loop

This is the part you actually memorise. Six commands.

### Step 1 — Start from a fresh `main`

```bash
git checkout main
git pull origin main
```

Do this **every time you sit down to work**, even if you worked an hour ago.
Someone may have merged something.

### Step 2 — Create your branch

```bash
git checkout -b punith/critic-early-stopping
```

`-b` means "create it and switch to it". Use it only when the branch is new.

To switch to a branch that already exists, drop the `-b`:

```bash
git checkout punith/critic-early-stopping
```

Confirm where you are at any time:

```bash
git branch
```

The `*` marks your current branch. If it's on `main`, stop and switch.

### Step 3 — Do your work, then check what changed

```bash
git status
```

Read this output before every commit. It shows you exactly what you're about to
save. If you see something surprising — a `.db` file, a whole `.venv`, a
notebook you didn't touch — sort that out before committing.

### Step 4 — Stage your changes

Stage only what belongs to this piece of work:

```bash
git add agent/loop.py agent/critic/predict.py
```

`git add .` stages everything, including files you forgot about. Use it only
when you have read `git status` and agree with all of it.

### Step 5 — Commit with a real message

```bash
git commit -m "punith: wire critic into the loop for early stopping"
```

Message format:

```
<your-name>: <what changed, lowercase, present tense>
```

**Good:**

```
punith: add threshold selection to the critic notebook
dhrub: return structured errors instead of raising
krishna: store episode summaries with a TTL
harish: fix the results table crashing on empty rows
```

**Bad:**

```
update              ← update what?
fix bug             ← which bug?
asdf                ← this is in the permanent record
Final version       ← it never is
changes as discussed in meeting   ← nobody remembers the meeting
```

A future teammate reading `git log` should understand the change without
opening it. That teammate is usually you, three weeks later.

### Step 6 — Push your branch

**First push on a new branch:**

```bash
git push -u origin punith/critic-early-stopping
```

`-u` links your local branch to the one on GitHub, so afterwards you can just
type:

```bash
git push
```

Push at the end of every working session, even if the work isn't finished.
Anything only on your laptop is one dead hard drive away from gone.

---

## Opening a pull request

When your piece of work is done and tested:

1. Go to <https://github.com/Punith-14/Agentic-Analyst>
2. GitHub shows a yellow banner: **"Compare & pull request"** — click it
3. Base: `main` ← Compare: your branch
4. **Title:** same style as a commit message — `punith: add trajectory critic`
5. **Description:** fill in this template

```markdown
## What this does
Two or three sentences in plain English.

## Why
What problem it solves.

## Files changed
- `agent/loop.py` — added the critic check before each step
- `agent/critic/predict.py` — new

## Testing
- [ ] `pytest -q` passes
- [ ] Ran end to end on 20 questions
- [ ] Does this change `contracts.py`? **No** / **Yes — I told the group on <date>**

## Anything reviewers should look at
The threshold is hardcoded at 0.694 for now — flagging it.
```

6. Under **Reviewers**, request **two** people: Punith, plus the owner of the
   layer your change affects downstream (see the table below). If it touches
   `contracts.py`, request **all three** others.
7. Post the PR link in the group chat.

---

## Who merges, and when

**Punith merges to `main`.** Nobody else presses the button.

This is deliberate. He owns `contracts.py`, and having one person read
everything that enters `main` is how integration problems get caught before they
become everyone's problem.

**But he has 2–3 hours a day and his own layer to build, so there is a deadline
attached:**

> Punith reviews within **24 hours**. If 24 hours pass and the PR already has
> an approval from the affected downstream layer, the author may merge it
> themselves.

That escape valve is part of the rule, not a workaround. A gate with nobody
promising a turnaround is what makes people stop opening PRs and start branching
off each other's unmerged branches.

### Who reviews what

Punith merges, but he is not the only reviewer — he doesn't work in your layer
and shouldn't be the only person checking your logic. Send the technical review
to whoever your change affects downstream:

| You are | Your reviewer | Why |
|---|---|---|
| Dhrub (A) | **Punith** | The loop calls his tools |
| Punith (B) | **Harish** | The orchestrator wraps the loop |
| Krishna (C) | **Harish** | The graph passes memory around |
| Harish (D) | **Punith** | Consumes the loop and the contracts |

So the flow is: **author opens PR → downstream owner reviews the code → Punith
reads it and merges.**

### How often to merge

**Three days maximum.** If a branch is older than three days, it is too big.

Time is the rough guide; **size** is the real one. A pull request should be
readable in about 15 minutes — past roughly 400 changed lines, reviewers stop
reading and start rubber-stamping, and then we have the process without the
benefit.

**Pushing and merging are different things, on different schedules:**

| | How often | Affects |
|---|---|---|
| `git push` | **Every day**, even mid-task, even broken | Nobody. It's a backup. |
| Merge to `main` | **Every 2–3 days**, when a piece works | Everyone |

Anything that exists only on your laptop is one dead hard drive away from gone.
Push daily regardless of whether the work is finished.

**If three days pass and the work isn't done:** merge the part that works and
carry the rest to a new branch. Half a feature merged beats a whole one stuck.

### Integrate early — this is the one that sinks student projects

The usual way a four-person project fails is not merge conflicts. It is four
people building alone for two weeks and finding out on day thirteen that the
pieces don't fit: the tools return a different shape than the loop expects, the
memory doesn't match what the graph passes in.

So in the **first three days**, everyone merges something skeletal to `main` —
stub functions that return the right *type* and nothing else. Dhrub's tool
returns an empty `ToolResult`. Krishna's memory returns an empty list. Then we
wire them together and run one question end to end.

It will do nothing useful. That is fine. The point is that the seams get tested
on day 3, when there is time to fix them.

`agent/_stubs.py` already does this for layer A — it is why the loop runs at all
without Dhrub's tools being finished. Same idea, applied across the team.

### After it's merged

Delete the branch (GitHub offers a button), then clean up locally:

```bash
git checkout main
git pull origin main
git branch -d punith/critic-early-stopping
```

---

## When things go wrong

### "Your branch is behind `main`" / PR shows conflicts

Someone merged while you were working. Bring their changes into your branch:

```bash
git checkout main
git pull origin main
git checkout punith/your-branch
git merge main
```

If git reports a conflict, it lists the affected files. Open each one and look
for:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```

Keep the correct code, delete all three marker lines, save. Then:

```bash
git add <the-file>
git commit -m "punith: merge main"
git push
```

**If the conflict is in a file you don't own, message the owner instead of
guessing.** They know which version is right; you don't.

### I committed to `main` by accident

Nothing is lost. Move the commit onto a branch:

```bash
git branch punith/rescue-my-work
git reset --hard origin/main
git checkout punith/rescue-my-work
```

Your work is now on `punith/rescue-my-work`, and `main` is clean.

### I committed something huge / secret

**Stop and tell the group before pushing.** Once it's on GitHub it's much harder
to remove, and a pushed API key must be treated as leaked and rotated.

If you haven't pushed yet:

```bash
git reset --soft HEAD~1     # undo the commit, keep your edits
```

Then fix the `.gitignore`, re-stage properly, and commit again.

### I want to throw away my uncommitted changes

```bash
git checkout -- <file>      # one file
git reset --hard            # everything — this cannot be undone
```

### I have no idea what state I'm in

```bash
git status
git log --oneline -10
git branch
```

Those three commands answer almost every "what's happening" question. Paste the
output into the group chat rather than experimenting.

---

## What never goes in the repo

The `.gitignore` blocks these. Don't work around it.

| Never commit | Why | Instead |
|---|---|---|
| `Data/spider_data/` | 1.8 GB | `python data_prep/prepare_spider.py` |
| `.venv/` | Thousands of files, machine-specific | `pip install -r requirements.txt` |
| `.env`, API keys | Security | `.env.example` is committed with blank values — copy it |
| `__pycache__/` | Generated | — |
| LLM weights (`.gguf`, `.safetensors`) | Hundreds of MB | Ollama manages these |

**What we *do* commit, deliberately.** Six things look like they should be
ignored and aren't. Don't "clean them up":

| Committed | Size | Why |
|---|---|---|
| `Data/db/*.db` | 11 MB | `chinook_1` is required by the tests, so CI fails without it. All three mean **nobody ever needs the 1.8 GB Spider download** |
| `Data/tasks/*.json` | 216 KB | Everyone must run the identical 150 questions, or our numbers aren't comparable across layers |
| `Data/trajectories/*.jsonl` | 10.2 MB | **Not reproducible.** Sampled at temperature 0.7 — those exact runs are gone forever if lost. Three of the four ML components train on them |
| `Data/critic/labelled_steps.parquet` | 889 KB | Derived, but it pins the exact table behind the reported numbers |
| `Data/critic/split.json` | 5.8 KB | The frozen train/test split. Must be identical for everyone or results drift |
| `models/critic_*.joblib` | 2.6 MB | So A, C and D can load the critic without retraining it |

**Do not merge the two trajectory files into one, and do not rename them.**

```
2026-08-15-train.jsonl         1,620 runs · 5,461 steps · 30.9% solve rate
2026-08-15-superseded.jsonl       80 runs ·   526 steps ·  9.2% solve rate
```

The superseded batch is excluded from training by **filename**
(`label_dataset.py --exclude superseded`), because every legacy run records the
same `context_policy` string — nothing inside the record distinguishes them.
Merge or rename, and 80 runs from a broken configuration silently rejoin the
training set with no way to separate them again.

### Notebooks

**The default rule: clear all outputs before committing.**

Jupyter/VS Code: Kernel → Restart & Clear All Outputs, then save. Outputs are
stored as large base64 blobs, which makes diffs unreadable and inflates the
repo.

**The exception, which is deliberate:** `notebooks/02_critic_models.ipynb` is
committed **with** its outputs. That notebook *is* the results record — the PR
curves, the confusion matrix, the feature importances. A mentor or teammate who
clones the repo should be able to see them without a GPU and a two-hour rerun.

So the real rule is: **strip outputs from notebooks you are working in, keep
them on notebooks that are the deliverable.** If you're unsure which yours is,
ask in the group chat.

Either way, if a notebook produces a chart the team needs to reference outside
the notebook, also save it as a `.png` in `docs/`.

---

## Automated tests (CI)

Every pull request automatically runs the test suite on GitHub. You'll see a
green tick or a red cross on your PR within about two minutes.

**A red cross blocks the merge.** Punith will not merge a failing PR, so fix it
before asking for review.

The workflow lives in `.github/workflows/tests.yml`. It installs
`requirements.txt` and runs `pytest -q` — no GPU, no Ollama, because the tests
use the scripted fake model.

**Why we bother:** four people share `contracts.py`. When Dhrub changes a tool
signature, it breaks Punith's loop. Without CI, that gets discovered two days
later by whoever pulls next. With CI, it gets discovered before the merge, by
the person who caused it.

**Run the same check locally before you push** — it's faster than waiting for
GitHub:

```bash
pytest -q
```

If a test fails and you believe the test is wrong rather than your code, say so
in the PR description. Don't delete the test quietly.

---

## Quick reference

```bash
# start work
git checkout main
git pull origin main
git checkout -b punith/short-description

# save work
git status
git add <files>
git commit -m "punith: what changed"
git push -u origin punith/short-description     # first push
git push                                    # after that

# catch up with main
git checkout main && git pull origin main
git checkout punith/short-description && git merge main

# where am I
git status
git branch
git log --oneline -10
```

---

## Questions

Ask in the group chat before running any command you don't understand. Nearly
everything in git is recoverable — but only if you ask *before* you try to fix
it yourself.
