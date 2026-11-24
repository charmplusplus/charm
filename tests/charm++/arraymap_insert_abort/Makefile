-include ../../common.mk
CHARMC = ../../../bin/charmc $(OPTS)

all: insert_abort

insert_abort: insert_abort.o
	$(CHARMC) -language charm++ -o $@ insert_abort.o

insert_abort.decl.h: insert_abort.ci
	$(CHARMC) insert_abort.ci

insert_abort.o: insert_abort.C insert_abort.decl.h
	$(CHARMC) -c insert_abort.C

ABORT_MSG = ckInsert without a PE is not supported with this array map

# Each run must exit nonzero and print ABORT_MSG. testrun skips (exit 0,
# "Skipping the test") when the process count does not divide the PE count;
# that counts as a pass. $(1): the -map option
define expect_abort
	@$(call run, ./insert_abort +p2 -map $(1) ) > abort-$(1).out 2>&1; rc=$$?; \
	if grep -q "Skipping the test" abort-$(1).out; then \
	  echo "insert_abort $(1) skipped"; \
	elif [ $$rc -eq 0 ]; then \
	  cat abort-$(1).out; echo "insert_abort $(1): expected a nonzero exit"; exit 1; \
	elif ! grep -q "$(ABORT_MSG)" abort-$(1).out; then \
	  cat abort-$(1).out; echo "insert_abort $(1): abort message missing"; exit 1; \
	else \
	  echo "insert_abort $(1) passed (exit $$rc)"; \
	fi
endef

test: all
	$(call expect_abort,cld)
	$(call expect_abort,prop)
	$(call expect_abort,propempty)
	$(call expect_abort,fast)
	$(call expect_abort,hilbert)

# tests/charm++/Makefile runs smptest in every directory on SMP builds; the
# runs above do not depend on the number of PEs per process.
smptest: all

clean:
	rm -f *.decl.h *.def.h *.o insert_abort insert_abort.exe charmrun abort-*.out
