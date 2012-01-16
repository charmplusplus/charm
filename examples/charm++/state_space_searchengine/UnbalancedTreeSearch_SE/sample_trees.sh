#-----------------------------------------------------------------------------#
# Sample UTS Workloads:
#
#  This file contains sample workloads for UTS, along with the tree statistics
#  for verifying correct output from the benchmark.  This file is intended to
#  be used in shell scripts or from the shell so that UTS can be run by:
#
#   $ source sample_workloads.sh
#   $ ./uts $T1
#
#-----------------------------------------------------------------------------#

# ====================================
# Small Workloads (~4 million nodes):
# ====================================

# (T1) Geometric [fixed] ------- Tree size = 4130071, tree depth = 10, num leaves = 3305118 (80.03%)
export T1="Mt 1 Ma 3 Md 10 Mb 4 Mr 19"

# (T5) Geometric [linear dec.] - Tree size = 4147582, tree depth = 20, num leaves = 2181318 (52.59%)
export T5="-t 1 -a 0 -d 20 -b 4 -r 34"

# (T2) Geometric [cyclic] ------ Tree size = 4117769, tree depth = 81, num leaves = 2342762 (56.89%)
export T2="-t 1 -a 2 -d 16 -b 6 -r 502"

# (T3) Binomial ---------------- Tree size = 4112897, tree depth = 1572, num leaves = 3599034 (87.51%)
export T3="Mt 0 Mb 2000 Mq 0.124875 Mm 8 Mr 42"

# (T4) Hybrid ------------------ Tree size = 4132453, tree depth = 134, num leaves = 3108986 (75.23%)
export T4="-t 2 -a 0 -d 16 -b 6 -r 1 -q 0.234375 -m 4 -r 1"

# ====================================
# Large Workloads (~100 million nodes):
# ====================================

# (T1L) Geometric [fixed] ------ Tree size = 102181082, tree depth = 13, num leaves = 81746377 (80.00%)
export T1L="Mt 1 Ma 3 Md 13 Mb 4 Mr 29"

# (T2L) Geometric [cyclic] ----- Tree size = 96793510, tree depth = 67, num leaves = 53791152 (55.57%)
export T2L="-t 1 -a 2 -d 23 -b 7 -r 220"

# (T3L) Binomial --------------- Tree size = 111345631, tree depth = 17844, num leaves = 89076904 (80.00%)
export T3L="-t 0 -b 2000 -q 0.200014 -m 5 -r 7"

# ====================================
# Extra Large (XL) Workloads (~1.6 billion nodes):
# ====================================

# (T1XL) Geometric [fixed] ----- Tree size = 1635119272, tree depth = 15, num leaves = 1308100063 (80.00%)
export T1XL="Mt 1 Ma 3 Md 15 Mb 4 Mr 29"

# ====================================
# Extra Extra Large (XXL) Workloads (~3-10 billion nodes):
# ====================================

# (T1XXL) Geometric [fixed] ---- Tree size = 4230646601, tree depth = 15 
export T1XXL="Mt 1 Ma 3 Md 15 Mb 4 Mr 19"

# (T3XXL) Binomial ------------- Tree size = 2793220501 
export T3XXL="Mt 0 Mb 2000 Mq 0.499995 Mm 2 Mr 316"

# (T2XXL) Binomial ------------- Tree size = 10612052303, tree depth = 216370, num leaves = 5306027151 (50.00%) 
export T2XXL="-t 0 -b 2000 -q 0.499999995 -m 2 -r 0"

# ====================================
# Wicked Large Workloads (~150-300 billion nodes):
# ====================================

# (T1WL) Geometric [fixed] ----- Tree size = 270751679750, tree depth = 18, num leaves = 216601257283 (80.00%)
export T1WL="-t 1 -a 3 -d 18 -b 4 -r 19"

# (T2WL) Binomial -------------- Tree size = 295393891003, tree depth = 1021239, num leaves = 147696946501 (50.00%)
export T2WL="-t 0 -b 2000 -q 0.4999999995 -m 2 -r 559"

# (T3WL) Binomial -------------- Tree size = T3WL: Tree size = 157063495159, tree depth = 758577, num leaves = 78531748579 (50.00%) 
export T3WL="-t 0 -b 2000 -q 0.4999995 -m 2 -r 559"

