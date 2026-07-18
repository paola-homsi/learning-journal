#!/usr/bin/env bash
# Create a new journal entry from a template.
# Usage:
#   ./scripts/new.sh paper "flashattention"     -> papers/2026-07-flashattention.md
#   ./scripts/new.sh experiment "lora-rank-sweep" -> experiments/2026-07-18-lora-rank-sweep.md
#   ./scripts/new.sh weekly                      -> weekly/2026-W29.md
#   ./scripts/new.sh project "04-production-rag" -> projects/04-production-rag/{DESIGN,DECISIONS,RETRO}.md
#   ./scripts/new.sh story "nan-loss-debugging"  -> notes/interview-stories/nan-loss-debugging.md
set -euo pipefail
cd "$(dirname "$0")/.."

TYPE="${1:?Usage: new.sh <paper|experiment|weekly|project|story> [name]}"
NAME="${2:-}"
slug() { echo "$1" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-'; }

case "$TYPE" in
  paper)
    [ -n "$NAME" ] || { echo "Name required"; exit 1; }
    OUT="papers/$(date +%Y-%m)-$(slug "$NAME").md" ;;
  experiment)
    [ -n "$NAME" ] || { echo "Name required"; exit 1; }
    OUT="experiments/$(date +%Y-%m-%d)-$(slug "$NAME").md" ;;
  weekly)
    OUT="weekly/$(date +%Y-W%V).md" ;;
  story)
    [ -n "$NAME" ] || { echo "Name required"; exit 1; }
    OUT="notes/interview-stories/$(slug "$NAME").md" ;;
  project)
    [ -n "$NAME" ] || { echo "Name required"; exit 1; }
    DIR="projects/$(slug "$NAME")"
    [ -e "$DIR" ] && { echo "Already exists: $DIR"; exit 1; }
    cp -r templates/project "$DIR"
    echo "Created $DIR/ (DESIGN.md, DECISIONS.md, RETRO.md)"
    exit 0 ;;
  *)
    echo "Unknown type: $TYPE"; exit 1 ;;
esac

[ -e "$OUT" ] && { echo "Already exists: $OUT"; exit 1; }
cp "templates/$TYPE.md" "$OUT"
# Stamp today's date into the new file where the template has a placeholder
sed -i "s/YYYY-MM-DD/$(date +%Y-%m-%d)/g" "$OUT"
echo "Created $OUT"
