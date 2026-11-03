#!/usr/bin/env bash
# Run the whole results pipeline: robust checkpoint selection (validation only), the benchmark on the unseen
# junctions, demo replays, static-site data, figures, README tables, GIF, video and screenshots.
#
#   PYTHON=.venv/bin/python [SKIP_SELECT=1] bash tools/finalize.sh [workers]      # log: results/finalize.log
#
# Screenshots need the static site served on port 8090 (python -m http.server 8090 --directory docs).
# Nothing is committed or published: review the results, then commit.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
W=${1:-10}
export PYTHONIOENCODING=utf-8
mkdir -p results
exec > >(tee -a results/finalize.log) 2>&1
step() { echo "== $(date '+%F %T') $*"; "$PY" "$@"; }
# SKIP_SELECT=1 keeps the policy already selected in models/ (e.g. one picked with tools/head_to_head.py)
if [ "${SKIP_SELECT:-0}" != "1" ]; then step -m chowkpilot.rl.select --revalidate 3 --ensemble --workers "$W"; fi
step -m chowkpilot.eval.benchmark --seeds 10 --workers "$W"
step -m chowkpilot.eval.make_replays --workers "$W"
step -m chowkpilot.eval.export_site
step -m chowkpilot.eval.figures
step -m chowkpilot.eval.readme_results
step -m chowkpilot.eval.make_gif --scenario hyd_hitech --baseline fixed
step -m chowkpilot.eval.make_video
if curl -s -o /dev/null http://localhost:8090/; then step tools/screenshots.py; else echo "skipped screenshots: no server on :8090"; fi
echo "== $(date '+%F %T') finalize done"
