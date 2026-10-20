#!/bin/bash
# after_chan4.sh <jobid>: the leftover minutes of the chan4 allocation. Installs the leanmd.ch that records the
# comm-graph edge at put() (so a put issued from a credit handler is charged to its sender), then four 8x4x4
# async arms, 100 steps each: control and -channels with the default Metis cut gate, then both with the gate
# off (+LBMetisMaxCutRise 1.0), so the two modes take the same kind of mapping and the step time difference is
# the channels' alone. Never sallocs, never scancels.
JID=$1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
cd $L
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); [ -n "$END_EPOCH" ] || { echo "no end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
cp /tmp/claude-54324/-u-bhosale-charm-reconverse/73524aa5-0acc-479f-ab89-3199163b9f8f/scratchpad/lmdbuild/leanmd.ch $L/leanmd.ch
echo "=== after_chan4 $(date +%T), left $(left_s) s, leanmd.ch $(ls -la --time-style=+%H:%M leanmd.ch | awk '{print $6}')"
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 TMO=90 LCIDEV=1 BIN=leanmd.ch
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
arm() {  # <LB> <channels> <tag> [extra LB flags]
  local lb=$1 ch=$2 tag=$3 extra="${4:-}"
  [ "$(left_s)" -gt 60 ] || { echo "skip $tag: $(left_s) s left"; return; }
  local TAG=${tag}_$(date +%H%M%S)
  env CHANNELS=$ch LB=$lb GRID="8 4 4" STEPS=100 PPN=4 GPP=1 PES=8 LMD_EXTRA="$extra" $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step' $R) | mean21-100 $(mean_steps $R 21 100) | mean42-100 $(mean_steps $R 42 100) | metis-moves $(grep -a 'Objects migrating' $R | sed -E 's/.*migrating: ([0-9]+).*/\1/' | tr '\n' ',') | cross-node $(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}') | ext/int MB $(grep -a 'External comm BEFORE' $R | head -1 | awk '{printf "%.0f", $5}')/$(grep -a 'Internal comm BEFORE' $R | head -1 | awk '{printf "%.0f", $5}') | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
  grep -a 'cross-group messages by host' $R | head -1 | cut -c1-160 | sed 's/^/    /'
}
arm async 0 ctl_async_8x4x4
arm async 1 ch_async_8x4x4
arm async 0 ctl_async_8x4x4_nogate "+LBMetisMaxCutRise 1.0"
arm async 1 ch_async_8x4x4_nogate "+LBMetisMaxCutRise 1.0"
echo "=== after_chan4 done $(date +%T), left $(left_s) s"
