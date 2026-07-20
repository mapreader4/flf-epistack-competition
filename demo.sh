#!/usr/bin/env bash
# One-command demo of the epistemic argument graph.
#
# Queries the PRE-BUILT two-document graph (eric_decision + will_decision) that ships in
# this repo. No API key, no model downloads, no rebuild -- it reads the committed
# artifacts and prints structured, provenance-carrying results.
#
#   bash demo.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# 1. Minimal environment: only the query-time deps (numpy, openai, python-dotenv).
#    Reuse an existing .venv that already has them; otherwise make a standard venv.
if [ -x .venv/bin/python ] && .venv/bin/python -c "import numpy, openai, dotenv" 2>/dev/null; then
  echo ">> using existing .venv"
else
  [ -x .venv/bin/python ] || { echo ">> creating virtual environment (.venv)"; python3 -m venv .venv; }
  echo ">> installing query dependencies"
  .venv/bin/python -m pip install -q --disable-pip-version-check numpy openai python-dotenv
fi

PY=.venv/bin/python
Q=scripts/query_epistemic_single_multidoc.py
DIR=artifacts/epistemic_2doc

echo
echo "=================================================================="
echo " Q1  Evidence for a hypothesis, aggregated ACROSS both judges"
echo "     H = 'an animal host at HSM was responsible for SARS-CoV-2'"
echo "=================================================================="
$PY "$Q" evidence-for n-00262 --data-dir "$DIR" --prefix "" --no-llm

echo
echo "=================================================================="
echo " Q2  Where the two judges (or the sides they cite) DISAGREE"
echo "=================================================================="
$PY "$Q" contested --data-dir "$DIR" --prefix "" --no-llm | head -c 2500
echo
echo
echo ">> done. (Add TOGETHER_API_KEY to .env and drop --no-llm for LLM-written prose answers.)"
