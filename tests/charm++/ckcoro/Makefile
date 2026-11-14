-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: ckcoro_tests names.o

ckcoro_tests: ckcoro_tests.C ckcoro_tests.decl.h basic.decl.h refnum.decl.h migration.decl.h owners.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 ckcoro_tests.C -o ckcoro_tests

# The two attribute words used as names (names.ci): compiled, not run.
names.o: names.C names.decl.h
	$(CHARMC) -c names.C -o names.o

%.decl.h: %.ci
	$(CHARMC) $<

# Projections events of resumed segments, on a build with tracing only
# (--enable-tracing provides libtrace-projections.a).
TRACELIB = $(wildcard ../../../lib/libtrace-projections.a)
ifneq ($(TRACELIB),)
all: tracetest
endif

tracetest: tracetest.C tracetest.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 tracetest.C -o tracetest -tracemode projections

# Begin events (type 2), at least: 4 elements x (1 invocation + 10 resumes)
# for run(), 4 if resumed segments are not traced; 4 x 10 for the label
# "work" and 4 for "setup". More when an entry runs inside a segment (a
# reduction's broadcast sent from contribute): Projections ends the
# segment's event and logs a second begin for it after the inner one.
# Exactly 4 for fan(), whose 100 tasks never wait and call nothing in the
# runtime (a task's first run is not a resume: 804 if it were traced).
# The files are named after the binary: tracetest.exe.sts on Windows.
trace-test: tracetest
	rm -rf trc
	mkdir trc
	$(call run, ./tracetest +p2 +traceroot trc)
	@for c in "run() 44" "work 40" "setup 4" "fan() 4 4"; do \
	  set -- $$c; \
	  ep=$$(awk -v n="\"$$1\"" '$$1 == "ENTRY" && $$4 == n {print $$3}' trc/*.sts); \
	  n=$$(gzip -dcf trc/*.log* | awk -v ep="$$ep" '$$1 == 2 && $$3 == ep {n++} END {print n+0}'); \
	  echo "tracetest: $$n begin events for $$1 (entry $$ep), expected at least $$2$${3:+, at most $$3}"; \
	  test -n "$$ep" && test "$$n" -ge "$$2" && { test -z "$$3" || test "$$n" -le "$$3"; } || exit 1; \
	done

clean:
	rm -f *.decl.h *.def.h *.o ckcoro_tests tracetest charmrun
	rm -rf log trc

test: all
	$(call run, ./ckcoro_tests +p1)
	$(call run, ./ckcoro_tests +p4)
	$(call run, ./ckcoro_tests +p4 +ck_fakemigrate)
	$(call run, ./ckcoro_tests +p2 +ck_fakemigrate_pup)
	$(call run, ./ckcoro_tests +p4 +ck_fakemigrate_pup)
	rm -rf log
	$(call run, ./ckcoro_tests -chkpt log +p4)
	-sync
	$(call run, ./ckcoro_tests +p4 +restart log)
ifneq ($(TRACELIB),)
	$(MAKE) trace-test
endif

.PHONY: all clean test trace-test
