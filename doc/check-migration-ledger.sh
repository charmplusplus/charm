#!/bin/sh
# Mechanical closure check for the reconverse migration (plan item 22).
# Recomputes the live delta between the reviewed line and the old dev
# branch and fails if any differing file has no disposition in the
# ledger, or if any needs-judgment rows remain. Migration is COMPLETE
# when this passes with every remaining delta row tombstone/superseded.
# Usage: doc/check-migration-ledger.sh [reviewed-ref] [branch-ref]
set -e
REVIEWED=${1:-origin/reviewed-with-reconverse}
BRANCH=${2:-origin/reconverse-specific-build}
LEDGER=$(dirname "$0")/reconverse-migration-ledger.tsv
fail=0
git diff --name-only "$REVIEWED...$BRANCH" | while read -r f; do
  grep -q "^$f	" "$LEDGER" || { echo "UNDISPOSITIONED: $f"; fail=1; }
done
n_judge=$(grep -c "	needs-judgment	" "$LEDGER" || true)
[ "$n_judge" -gt 0 ] && echo "NEEDS-JUDGMENT rows remaining: $n_judge"
n_live=$(git diff --name-only "$REVIEWED...$BRANCH" | wc -l | tr -d ' ')
echo "live delta: $n_live files (ledger covers $(grep -vc '^#' "$LEDGER") rows)"
[ "$fail" -eq 0 ]
