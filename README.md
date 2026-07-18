# AI Engineering Learning Journal

My working log on the path from Senior Software Engineer → Senior AI Engineer.
Every paper I read, every experiment I run, every project I build, every week I learn — captured here.

> **Why this repo exists:** Interview answers, blog posts, and portfolio writeups aren't written from memory — they're written from notes taken *at the moment of learning*. This repo is the raw material. By interview time, it becomes a searchable database of "tell me about a time..." stories, architecture decisions, and paper discussions.

## Structure

```
learning-journal/
├── papers/                  # One note per paper (3-pass method) — papers/YYYY-MM-title.md
├── experiments/             # One note per experiment — experiments/YYYY-MM-DD-name.md
├── projects/                # One folder per portfolio project (design doc + decisions + retro)
│   └── 01-llm-chatbot/
│       ├── DESIGN.md        # Problem, architecture, tradeoffs (written BEFORE building)
│       ├── DECISIONS.md     # Running log of choices + why (written DURING)
│       └── RETRO.md         # What worked, what broke, what I'd change (written AFTER)
├── weekly/                  # Weekly review — weekly/YYYY-WNN.md
├── notes/
│   ├── flashcards/          # Anki-exportable Q&A (one topic per file)
│   └── interview-stories/   # STAR stories harvested from projects & experiments
├── templates/               # Blank templates for all of the above
├── scripts/
│   └── new.sh               # Create a new entry from a template: ./scripts/new.sh paper "flashattention"
└── PROGRESS.md              # The 310-item roadmap checklist, tracked here
```

## The workflow (keep it lightweight — 15 min/day max)

1. **Learning something?** No note needed for passive watching/reading — but the moment you *implement*, open an experiment note.
2. **Reading a paper (Pass 2)?** `./scripts/new.sh paper "lora"` → fill the template as you read, not after.
3. **Starting a project?** Copy `templates/project/` into `projects/NN-name/`. Write `DESIGN.md` *first* — the act of writing it is the design work.
4. **Made a non-obvious choice?** One line in the project's `DECISIONS.md`. These lines become interview answers verbatim.
5. **Something broke?** That's the most valuable entry of all. Log it in the experiment note or `DECISIONS.md`. Interviewers ask about failures more than successes.
6. **Sunday (20 min):** `./scripts/new.sh weekly` → fill the weekly review. Update `PROGRESS.md`. Move anything story-worthy into `notes/interview-stories/`.

## Rules that make this compound

- **Write at the moment, not later.** A rough note today beats a polished note never.
- **Every entry answers "so what?"** — what did I learn that I didn't know before?
- **Numbers over adjectives.** "Reranker improved recall@5 from 0.61 → 0.78" not "reranker helped a lot."
- **Failures get logged.** NaN losses, bad chunking, agents that looped — these are interview gold.
- **Public by default** (or a public mirror of the non-sensitive parts). This repo *is* credibility.
