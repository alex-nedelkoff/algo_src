#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # project root

RESULTS_FILE="autoresearch/results.tsv"
DIFFS_DIR="autoresearch/diffs"
TRAIN_PY="autoresearch/train.py"
TRAIN_PY_ORIG="autoresearch/train.py.orig"

mkdir -p "$DIFFS_DIR"

# Save original train.py for diffing
if [ ! -f "$TRAIN_PY_ORIG" ]; then
    cp "$TRAIN_PY" "$TRAIN_PY_ORIG"
fi

# Initialize results.tsv with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    printf "exp_id\ttimestamp\tscore\tavg_gates\tcrash_rate\talt_std\tavg_steps\tmax_gates\tseed\tlambda_gate\tlambda_prog\tlambda_rate\tlambda_offset\tlambda_perc\tlambda_delta_u\tlambda_crash\tlambda_alive\tv_max\tcorner_noise_k\tcorner_dropout_onset\tlearning_rate\tent_coef\tclip_range\tgae_lambda\tgamma\tdr_percentage\n" > "$RESULTS_FILE"
fi

# Determine next exp_id
get_next_exp_id() {
    local last_id
    last_id=$(tail -n 1 "$RESULTS_FILE" 2>/dev/null | cut -f1)
    if [ "$last_id" = "exp_id" ] || [ -z "$last_id" ]; then
        echo 0
    else
        echo $((last_id + 1))
    fi
}

# Extract all 4 parameter dicts from train.py via AST parsing.
# Outputs 17 tab-separated values matching TSV columns 10-26.
extract_params() {
    python -c "
import ast

with open('$TRAIN_PY') as f:
    tree = ast.parse(f.read())

dicts = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                'REWARD_WEIGHTS', 'EKF_PARAMS', 'TRAINING_PARAMS', 'DOMAIN_RAND'
            ):
                dicts[target.id] = ast.literal_eval(node.value)

rw = dicts['REWARD_WEIGHTS']
ekf = dicts['EKF_PARAMS']
tp = dicts['TRAINING_PARAMS']
dr = dicts['DOMAIN_RAND']

vals = [
    rw['lambda_gate'], rw['lambda_prog'], rw['lambda_rate'],
    rw['lambda_offset'], rw['lambda_perc'], rw['lambda_delta_u'],
    rw['lambda_crash'], rw['lambda_alive'], rw['v_max'],
    ekf['corner_noise_k'], ekf.get('corner_dropout_onset', 'None'),
    tp['learning_rate'], tp['ent_coef'], tp['clip_range'],
    tp['gae_lambda'], tp['gamma'],
    dr['percentage'],
]
print('\t'.join(str(v) for v in vals))
"
}

# Count consecutive diverged experiments (crash_rate >= 0.9)
count_consecutive_diverged() {
    python -c "
import csv
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f, delimiter='\t')
    count = 0
    for row in reader:
        if float(row['crash_rate']) >= 0.9:
            count += 1
        else:
            count = 0
    print(count)
"
}

# Check for score plateau (no improvement in last 10 experiments)
check_plateau() {
    local n_exps
    n_exps=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    if [ "$n_exps" -lt 10 ]; then
        echo "no"
        return
    fi
    python -c "
import csv
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f, delimiter='\t')
    scores = [float(row['score']) for row in reader]
if len(scores) < 10:
    print('no')
else:
    best_before = max(scores[:-10]) if len(scores) > 10 else -999
    best_recent = max(scores[-10:])
    print('yes' if best_recent <= best_before else 'no')
"
}

MAX_CONSECUTIVE_DIVERGED=3

# Verify dependencies
python --version >/dev/null 2>&1 || { echo "python not found — activate conda env first"; exit 1; }
claude --version >/dev/null 2>&1 || { echo "Claude CLI not found. Install from https://claude.ai/claude-code"; exit 1; }

echo "=== Autoresearch EKF+PPO Tuning Loop ==="
echo "18 parameters: 9 reward + 2 EKF + 5 training + 1 DR"
echo "Score = avg_gates - 2 * crash_rate"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    EXP_ID=$(get_next_exp_id)
    SEED=$((42 + EXP_ID))
    TIMESTAMP=$(date -Iseconds)

    echo "--- Experiment $EXP_ID (seed=$SEED) ---"

    # Save diff
    diff -u "$TRAIN_PY_ORIG" "$TRAIN_PY" > "$DIFFS_DIR/exp_$(printf '%03d' "$EXP_ID").diff" || true

    # Extract current parameters
    PARAMS=$(extract_params)

    # Run training + eval
    echo "Running training ($EXP_ID)..."
    RESULTS_JSON=$(PYTHONPATH=. python autoresearch/train.py "$EXP_ID" 2>&1 | tail -n 1)

    # Parse JSON results
    SCORE=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['score'])")
    AVG_GATES=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['avg_gates'])")
    CRASH_RATE=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['crash_rate'])")
    ALT_STD=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['alt_std'])")
    AVG_STEPS=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['avg_steps'])")
    MAX_GATES=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['max_gates'])")

    # Append to results.tsv — $PARAMS contains embedded tabs for columns 10-26
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$EXP_ID" "$TIMESTAMP" "$SCORE" "$AVG_GATES" "$CRASH_RATE" "$ALT_STD" "$AVG_STEPS" "$MAX_GATES" "$SEED" "$PARAMS" \
        >> "$RESULTS_FILE"

    echo "Score: $SCORE | Gates: $AVG_GATES | Crash: $CRASH_RATE | Alt: $ALT_STD"

    # Check divergence
    DIVERGED=$(count_consecutive_diverged)
    if [ "$DIVERGED" -ge "$MAX_CONSECUTIVE_DIVERGED" ]; then
        echo ""
        echo "!!! $DIVERGED consecutive diverged experiments (crash_rate >= 0.9). Pausing."
        echo "Review results.tsv and adjust bounds or strategy before restarting."
        exit 1
    fi

    # Check plateau
    PLATEAU=$(check_plateau)
    if [ "$PLATEAU" = "yes" ]; then
        echo ""
        echo "Score plateau detected: no improvement in last 10 experiments."
        echo "Best score: $(python -c "
import csv
with open('$RESULTS_FILE') as f:
    scores = [float(row['score']) for row in csv.DictReader(f, delimiter='\t')]
print(f'{max(scores):.4f}')
")"
        echo "Pausing. Review results.tsv for insights."
        exit 0
    fi

    # Invoke LLM agent to edit train.py
    echo ""
    echo "Invoking LLM agent to propose next experiment..."
    claude -p "$(cat <<PROMPT
You are an autonomous parameter tuning agent for drone racing RL with EKF.

Read the research directives:
$(cat autoresearch/program.md)

Here are all experiment results so far:
$(cat "$RESULTS_FILE")

Your task: edit the 4 parameter dicts (REWARD_WEIGHTS, EKF_PARAMS,
TRAINING_PARAMS, DOMAIN_RAND) in autoresearch/train.py.
Stay within the parameter bounds. Change 1-3 params per experiment
to isolate effects. Look for trends in the results before choosing
what to change.

$(python -c "print('WARNING: The last experiment DIVERGED (crash_rate=$CRASH_RATE). Try a less aggressive change.' if float('$CRASH_RATE') >= 0.9 else '')")

Current train.py:
$(cat "$TRAIN_PY")

Edit the parameter dict values only. Do not touch anything below the FIXED line.
PROMPT
)" --allowedTools Edit

    echo ""
done
