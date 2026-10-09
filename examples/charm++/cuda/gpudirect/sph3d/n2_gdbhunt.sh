#!/bin/bash
# n2_gdbhunt.sh <jobid>: async arms in the crashing configuration (256 patches, sort/100, Metis alone, floor 3%, cut 10%,
# period 25 window 10, 32 GB pool) under gdb -batch; a fault prints every thread's backtrace into logs/gdb_<tag>/.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export NODES=2 NP=8 export POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 SPH_WRAP=$D/numa_wrap_sph_gdb.sh
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
i=0
while [ "$(left_s)" -gt 240 ]; do
  i=$((i+1)); TAG=gdb${i}_p25_asyncT; export SPH_GDB_DIR=$D/logs/gdb_$TAG; mkdir -p $SPH_GDB_DIR
  DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 LB=async LBDEBUG=1 TMO=400 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 +LBMetisHostByMsgs $XF > /dev/null 2>&1
  L=$D/logs/$TAG.log
  ms=$(grep -E "^  step +[0-9]+:" $L | awk '{for(k=1;k<=NF;k++) if($k=="step") st=$(k+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,m); if(st>1000){s+=m[1];n++}} END{if(n) printf "%.1f", s/n; else print "NA"}')
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms, 2nd-half ${ms} | faults: $(grep -l 'signal caught' $SPH_GDB_DIR/*.log 2>/dev/null | wc -l) rank log(s) with a backtrace"
  for g in $(grep -l "signal caught" $SPH_GDB_DIR/*.log 2>/dev/null); do echo "=== $g"; grep -A 18 "signal caught" $g | head -24 | cut -c1-200; done
done
echo "gdbhunt done $(date +%T), left $(squeue -j $JID -h -o %L)"
