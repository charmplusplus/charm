#!/bin/bash
# Builds the two ablation variants of leanmd (senders not told; one migration stream) in a work dir with
# CHARM_DIR overridden, under their own names; the standard binaries are not touched. Ends by restoring the
# final source and rebuilding the runtime libraries.
B=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda
R=/u/bhosale/charm-reconverse
G=$R/examples/charm++/cuda/gpudirect
W=$G/leanmd/logs/vbuild
export LD_LIBRARY_PATH=/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
cd $R || exit 1
cp -p $G/leanmd/*.cc $G/leanmd/*.h $G/leanmd/*.cu $G/leanmd/*.ci $G/leanmd/Makefile $W/
build_lmd() {  # <name>
  ( cd $W && rm -f Main.o Cell.o Compute.o leanmdCUDA.o leanmd leanmd.decl.h leanmd.def.h && make CHARM_DIR=$R 2>&1 | grep -iE 'undefined| error|error:|Error [0-9]' | head -5
    [ -x leanmd ] && cp -p leanmd $G/leanmd/$1 && echo "built $1 $(date +%T)" || echo "FAILED $1" )
}
echo "=== variant 1: no sender notification $(date +%T)"
cp -p src/ck-core/cklocation.C $W/cklocation.C.final
python3 -I - <<'PY'
p='/u/bhosale/charm-reconverse/src/ck-core/cklocation.C'
s=open(p).read()
old='''void CkLocMgr::informSenders(const CkArrayIndex& idx, CmiUInt8 id, const int* pes, int n, int toPe)
{
  if (n == 0) return;
'''
assert old in s
s=s.replace(old, old+'  return;   // ABLATION VARIANT: senders not told\n',1)
open(p,'w').write(s)
PY
make -C $B -j12 ck 2>&1 | grep -iE ' error |error:' | head -5
build_lmd leanmd.noinform
cp -p $W/cklocation.C.final src/ck-core/cklocation.C
echo "=== variant 2: one migration stream $(date +%T)"
cp -p src/arch/cuda/hybridAPI/devicemanager.h $W/devicemanager.h.final
sed -i 's/enum { MIGRATION_STREAMS = 4 };/enum { MIGRATION_STREAMS = 1 };/' src/arch/cuda/hybridAPI/devicemanager.h
make -C $B -j12 hybridapi ck 2>&1 | grep -iE ' error |error:' | head -5
build_lmd leanmd.onestream
cp -p $W/devicemanager.h.final src/arch/cuda/hybridAPI/devicemanager.h
echo "=== restore the final runtime $(date +%T)"
make -C $B -j12 hybridapi ck 2>&1 | grep -iE ' error |error:' | head -5
make -C $B -j8 moduleEveryLB moduleDiffusionLB moduleCommonLBs moduleMetisLB 2>&1 | grep -iE ' error |error:' | head -5
echo "final source: ablation lines $(grep -c 'ABLATION VARIANT' src/ck-core/cklocation.C), $(grep -o 'MIGRATION_STREAMS = [0-9]' src/arch/cuda/hybridAPI/devicemanager.h)"
echo "=== variants done $(date +%T)"
