#!/bin/bash
#SBATCH --job-name=scale
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Weak/strong scaling for leanmd AND sph2d, one job per node count. The node
# count is NOT fixed here: submit with `sbatch -N <n>`, or let submit_scale.sh
# do it for a whole sweep. The job runs scale.sh with its OWN node count, so
# every srun spans the entire allocation -- nothing idles.
#
#   sbatch -N 2 --time=00:40:00 scale.sbatch                   # everything, 2 nodes
#   sbatch -N 4 --export=ALL,KINDS=weak scale.sbatch           # weak only
#   bash submit_scale.sh "1 2 4 8" leanmd weak                 # the whole sweep
#
# --export=ALL,VAR=... passes APPS / KINDS / ARMS / STEPS / PERIOD / LAG /
# LMD_GRID / COSTCFG straight through to scale.sh.
#
# --exclusive and 16 cpus-per-task take the WHOLE node (4 x 16 cores, one NUMA
# domain per rank, ThreadsPerCore=1). It is free: TRESBillingWeights on
# gpuA40x4 is CPU=31.25, GRES/gpu=500, and a 4-GPU job already bills 4000 =
# 64 x 31.25 + 4 x 500 whether 32 or 64 cores were requested. What it buys is
# real: the wrappers pin the PEs to the first 8 cores of each domain, so cores
# 8-15 are then free for the runtime's helper threads (4 cuda-EvtHandlr, the
# driver thread, comm) instead of those contending with the PEs -- which is
# exactly the untested hypothesis for moe's PPN-8 output-exchange stall. It
# also removes any chance of the +pemap target falling outside the job's
# cpuset. NOTE: every number in the notes so far was measured at 8
# cpus-per-task; keep a sweep self-consistent rather than mixing the two.
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
# Stop launching arms 3 minutes before the wall, so the summary still prints.
LIMIT_MIN=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null | grep -oE ' TimeLimit=[0-9:-]+' | cut -d= -f2 | awk -F'[-:]' '{if(NF==4)print $1*1440+$2*60+$3; else print $1*60+$2}')
export DEADLINE_MIN=${DEADLINE_MIN:-$(( ${LIMIT_MIN:-60} - 3 ))}
echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s) [$SLURM_JOB_NODELIST], limit ${LIMIT_MIN}min, deadline ${DEADLINE_MIN}min"
exec bash $ROOT/scale.sh "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "${APPS:-leanmd sph2d}" "${KINDS:-weak strong}" "${ARMS:-nolb sync async}"
