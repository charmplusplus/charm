#!/bin/bash
# after_stop1.sh: if n2_fix1.sh stops on a hang of the fixed write build, install the probe3 build (process-wide
# owner search, location-manager view, silent-PE check) and run up to 2 arms on the kept node.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; S=/tmp/claude-54324/-u-bhosale-charm-reconverse/73524aa5-0acc-479f-ab89-3199163b9f8f/scratchpad
JID=$1; cd $L; ulimit -c 0
cp $S/lmdbuild/leanmd $L/leanmd.new && mv -f $L/leanmd.new $L/leanmd && echo "probe3 build installed $(date +%T)"
sed -e 's/BIN=leanmd.pull3/BIN=leanmd/; s/pull3_ch_async/probe3_ch_async/; s/=== pull2 done/=== stop1 done/' $L/after_pull2.sh > $L/after_stop1_arms.sh
bash $L/after_stop1_arms.sh $JID
