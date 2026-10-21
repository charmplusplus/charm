#!/bin/bash
# Starts the ablation chain once the band chain is done and both variant binaries exist.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
until grep -q 'band done' $L/logs/stable2_chain.out 2>/dev/null; do sleep 10; done
until grep -q 'variants done' $L/logs/vbuild.log 2>/dev/null; do sleep 10; done
if [ -x $L/leanmd.noinform ] && [ -x $L/leanmd.onestream ]; then
  setsid -f nohup bash $L/logs/stable3_chain.sh 23047794 > $L/logs/stable3_chain.out 2>&1 < /dev/null
  echo "ablation chain started $(date +%T)"
else
  echo "variants missing $(date +%T)"; echo "=== stable3 ablation done (not run)" > $L/logs/stable3_chain.out
fi
