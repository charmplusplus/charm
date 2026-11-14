-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

# Found from this Makefile's real location, so a build directory's copy (a
# symlink into the source tree) finds the model files too.
RF_MODEL ?= $(dir $(realpath $(firstword $(MAKEFILE_LIST))))../../../../src/ck-ldb/rf_model
# The model reader needs zlib; builds without it (Windows) skip the model run.
HAVE_ZLIB := $(shell grep -c '^\#define CMK_USE_ZLIB 1' ../../../../include/conv-autoconfig.h 2>/dev/null)

all: period_selection

period_selection: period_selection.decl.h period_selection.C
	$(CHARMC) period_selection.C -o period_selection -module TreeLB

period_selection.decl.h: period_selection.ci
	$(CHARMC) period_selection.ci

# The model line has MetaBalancer pick the strategy with its random forest
# model (src/ck-ldb/rf_model).
test: period_selection
	$(call run, +p1 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
	$(call run, +p2 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
	$(call run, +p4 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)
ifeq ($(HAVE_ZLIB),1)
	$(call run, +p4 ./period_selection -anylb +MetaLB +MetaLBModelDir $(RF_MODEL) +LBObjOnly)
else
	@echo "meta_lb_test: skipping the random forest model run: Charm++ was built without zlib"
endif

testp: period_selection
	$(call run, +p$(P) ./period_selection +balancer RotateLB +MetaLB +LBObjOnly)

smptest: period_selection
	$(call run, +p2 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly ++ppn 2)
	$(call run, +p4 ./period_selection +balancer RotateLB +MetaLB +LBObjOnly ++ppn 2)

clean:
	rm -rf *.decl.h *.def.h period_selection charmrun

