-include ../../common.mk
CHARMDIR = ../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)
# The build directory: the tree's bin is a link into it.
BUILDDIR = $(realpath $(CHARMDIR)/bin/..)

OBJS = driver.o side.o converse_cases.o plain.o zerocopy.o direct.o

all: commbench

commbench: $(OBJS)
	$(CHARMC) -language charm++ -o commbench $(OBJS)

commbench.decl.h commbench.def.h: commbench.ci
	$(CHARMC) commbench.ci

# Commits and the build target, embedded in every run record. Rewritten on
# each make, but only touched when the contents change.
build_info.h: FORCE
	@{ \
	  echo '#define COMMBENCH_CHARM_COMMIT "'$$(git -C $(CHARMDIR) rev-parse --short=9 HEAD 2>/dev/null || echo unknown)'"'; \
	  echo '#define COMMBENCH_CHARM_DIRTY '$$(git -C $(CHARMDIR) diff --quiet HEAD -- src contrib 2>/dev/null && echo 0 || echo 1); \
	  echo '#define COMMBENCH_RECONVERSE_COMMIT "'$$(git -C $(CHARMDIR)/contrib/reconverse rev-parse --short=7 HEAD 2>/dev/null || echo unknown)'"'; \
	  echo '#define COMMBENCH_LCI_COMMIT "'$$(git -C $(BUILDDIR)/_deps/lci-src rev-parse --short=8 HEAD 2>/dev/null || echo unknown)'"'; \
	  echo '#define COMMBENCH_BUILD_TARGET "'$(notdir $(BUILDDIR))'"'; \
	  echo '#define COMMBENCH_LCI_PACKET_SIZE '$$(sed -n 's/^LCI_PACKET_SIZE_DEFAULT:STRING=//p' $(BUILDDIR)/CMakeCache.txt 2>/dev/null | grep . || echo 0); \
	  echo '#define COMMBENCH_LCI_WITH_SHM '$$(sed -n 's/^LCI_WITH_SHM:BOOL=//p' $(BUILDDIR)/CMakeCache.txt 2>/dev/null | grep -qix on && echo 1 || echo 0); \
	} > build_info.h.tmp
	@cmp -s build_info.h.tmp build_info.h || mv build_info.h.tmp build_info.h
	@rm -f build_info.h.tmp

FORCE:

HEADERS = commbench.h commbench.decl.h

driver.o: driver.C endpoints.h build_info.h commbench.def.h $(HEADERS)
	$(CHARMC) -c driver.C

side.o: side.C build_info.h $(HEADERS)
	$(CHARMC) -c side.C

converse_cases.o: converse_cases.C $(HEADERS)
	$(CHARMC) -c converse_cases.C

plain.o: cases/plain.C $(HEADERS)
	$(CHARMC) -I. -c cases/plain.C -o plain.o

zerocopy.o: cases/zerocopy.C $(HEADERS)
	$(CHARMC) -I. -c cases/zerocopy.C -o zerocopy.o

direct.o: cases/direct.C $(HEADERS)
	$(CHARMC) -I. -c cases/direct.C -o direct.o

# A short smoke run: every case, two sizes, one repeat. (No commas inside
# $(call run, ...): make would split the argument there.)
test: all
	$(call run, ./commbench +p2 --sizes 8 --extra-sizes 8097 --reps 1 --sample-time 0.001 --out test-run.jsonl)

clean:
	rm -f *.decl.h *.def.h *.o commbench charmrun build_info.h build_info.h.tmp test-run.jsonl

.PHONY: all test clean FORCE
