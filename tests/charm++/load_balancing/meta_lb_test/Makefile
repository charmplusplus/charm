-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

RF_MODEL = ../../../../src/ck-ldb/rf_model

all: period_selection

period_selection: period_selection.decl.h period_selection.C
	$(CHARMC) period_selection.C -o period_selection -module TreeLB

period_selection.decl.h: period_selection.ci
	$(CHARMC) period_selection.ci

# The model line has MetaBalancer pick the strategy with its random forest
# model (src/ck-ldb/rf_model). It is off by default: LBManager::switchLoadbalancer
# builds the TreeLB configuration with the keys "Root" and "Process", and
# TreeLB requires "root" and "process", so the first switch aborts ("TreeLB:
# Configuration must include "process" section."). Set META_LB_MODEL=1 to run it.
test: period_selection
	$(call run, +p1 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
	$(call run, +p2 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
	$(call run, +p4 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
ifeq ($(META_LB_MODEL),1)
	$(call run, +p4 ./period_selection -anylb +MetaLB +MetaLBModelDir $(RF_MODEL) +LBObjOnly)
endif

testp: period_selection
	$(call run, +p$(P) ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)

smptest: period_selection
	$(call run, +p2 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly ++ppn 2)
	$(call run, +p4 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly ++ppn 2)

clean:
	rm -rf *.decl.h *.def.h period_selection charmrun

