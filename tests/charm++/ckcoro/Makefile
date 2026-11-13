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

# At least 4 elements x (1 invocation + 10 resumes) begin events (type 2) for
# run(); 4 if resumed segments are not traced. More when an entry runs inside
# a segment (a reduction's broadcast sent from contribute): Projections ends
# the segment's event and logs a second begin for it after the inner one.
# The files are named after the binary: tracetest.exe.sts on Windows.
trace-test: tracetest
	rm -rf trc
	mkdir trc
	$(call run, ./tracetest +p2 +traceroot trc)
	@ep=$$(awk '/^ENTRY CHARE .* "run\(\)"/ {print $$3}' trc/*.sts); \
	n=$$(gzip -dcf trc/*.log* | awk -v ep=$$ep '$$1 == 2 && $$3 == ep {n++} END {print n+0}'); \
	echo "tracetest: $$n begin events for run() (entry $$ep), expected at least 44"; test "$$n" -ge 44

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
