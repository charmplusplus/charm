-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)
MPICXX ?= mpicxx
# LAUNCH must be the launcher the MPI library requires, starting NP
# processes, e.g. LAUNCH="srun --mpi=cray_shasta -n 2" for Cray MPICH.
NP ?= 2
LAUNCH ?= mpirun -np $(NP)
PES ?= 2
# Bounds each testexit run where the timeout command exists.
TIMEOUT ?= $(shell command -v timeout >/dev/null 2>&1 && echo "timeout 120")

all: mpi_interop mpi_exitcode

phaselib.decl.h: phaselib.ci
	$(CHARMC) $<

phaselib.o: phaselib.C phaselib.decl.h phaselib.h
	$(CHARMC) -c phaselib.C

libmodulephaselib.a: phaselib.o
	$(CHARMC) -o $@ phaselib.o

mpi_interop.o: mpi_interop.cpp phaselib.h
	$(CHARMC) -c++ $(MPICXX) -c $< -o $@

mpi_interop: mpi_interop.o libmodulephaselib.a
	$(CHARMC) -ld++ $(MPICXX) -mpi -nomain-module -o $@ mpi_interop.o -L. -module phaselib

exitcode.decl.h: exitcode.ci
	$(CHARMC) $<

exitcode.o: exitcode.C exitcode.decl.h
	$(CHARMC) -c exitcode.C

mpi_exitcode.o: mpi_exitcode.cpp
	$(CHARMC) -c++ $(MPICXX) -c $< -o $@

# The main module is exitcode.o's, so no -nomain-module.
mpi_exitcode: mpi_exitcode.o exitcode.o
	$(CHARMC) -ld++ $(MPICXX) -mpi -o $@ mpi_exitcode.o exitcode.o

test: all
	$(LAUNCH) ./mpi_interop $(NP) +pe $(PES)
	@$(MAKE) --no-print-directory testabort
	@$(MAKE) --no-print-directory testexit

# A CkAbort on the last PE must end the whole job through MPI_Abort, the
# abort hook CharmLibInit registers (reconverse #321): the run must fail, show
# the abort, and show the hook's line saying it calls MPI_Abort (the MPI
# library's own report is not required: Open MPI's mpirun drops it at times).
# Without the hook only the aborting process ends; mpirun then ends the
# others, but srun without --kill-on-bad-exit does not.
testabort: all
	@$(LAUNCH) ./mpi_interop $(NP) abort +pe $(PES) > abort.out 2>&1; rc=$$?; \
	cat abort.out; \
	if [ $$rc -eq 0 ]; then echo "testabort: expected a nonzero exit"; exit 1; fi; \
	if ! grep -q "mpi_interop: deliberate abort" abort.out; then \
	  echo "testabort: the abort did not happen"; exit 1; fi; \
	if ! grep -q "CharmLibInit: ending the MPI job with MPI_Abort" abort.out; then \
	  echo "testabort: the job did not end through CharmLibInit's abort hook"; exit 1; fi; \
	echo "mpi_interop testabort passed"

# A mainchare program that runs inside CharmLibInit and calls CkExit(code) on
# the last PE: every rank must get code from CharmLibExit and the launcher
# must exit with it, for code 0 and for a nonzero code. The defect this
# guards against is a hang in CharmLibExit, hence TIMEOUT (exit status 124).
testexit: mpi_exitcode
	@for code in 0 3; do \
	  $(TIMEOUT) $(LAUNCH) ./mpi_exitcode $$code +pe $(PES) > exit.out 2>&1; rc=$$?; \
	  cat exit.out; \
	  if [ -n "$(TIMEOUT)" ] && [ $$rc -eq 124 ]; then \
	    echo "testexit: the run with code $$code did not end within the time limit ($(TIMEOUT))"; exit 1; fi; \
	  if [ $$rc -ne $$code ]; then \
	    echo "testexit: launcher exit status $$rc, expected $$code"; exit 1; fi; \
	  n=`grep -c "^rank [0-9]* exit code $$code$$" exit.out`; \
	  if [ $$n -ne $(NP) ]; then \
	    echo "testexit: $$n of $(NP) ranks reported exit code $$code"; exit 1; fi; \
	done; \
	echo "mpi_exitcode testexit passed"

clean:
	rm -f mpi_interop mpi_exitcode *.o *.a *.decl.h *.def.h charmrun charm_all_libs.sh abort.out exit.out
