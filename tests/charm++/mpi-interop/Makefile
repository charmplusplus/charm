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

all: mpi_interop mpi_exitcode mpi_main_exit mpi_main_exit_charmc

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

# exitcode.C calls MPI.
exitcode.o: exitcode.C exitcode.decl.h
	$(CHARMC) -c++ $(MPICXX) -c exitcode.C -o $@

mpi_exitcode.o: mpi_exitcode.cpp
	$(CHARMC) -c++ $(MPICXX) -c $< -o $@

# The main module is exitcode.o's, so no -nomain-module.
mpi_exitcode: mpi_exitcode.o exitcode.o
	$(CHARMC) -ld++ $(MPICXX) -mpi -o $@ mpi_exitcode.o exitcode.o

# Main mode: the same program as an ordinary Charm++ run inside the MPI job,
# with a main that calls CharmMpiMain ...
mpi_main.o: mpi_main.cpp
	$(CHARMC) -c++ $(MPICXX) -c $< -o $@

mpi_main_exit: mpi_main.o exitcode.o
	$(CHARMC) -ld++ $(MPICXX) -nomain -o $@ mpi_main.o exitcode.o

# ... and with the main that charmc -mpi-main supplies.
mpi_main_exit_charmc: exitcode.o
	$(CHARMC) -ld++ $(MPICXX) -mpi-main -o $@ exitcode.o

test: all
	$(LAUNCH) ./mpi_interop $(NP) +pe $(PES)
	@$(MAKE) --no-print-directory testabort
	@$(MAKE) --no-print-directory testexit
	@$(MAKE) --no-print-directory testmain
	@$(MAKE) --no-print-directory testmainabort

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
	if ! grep -q "MPI interop: ending the MPI job with MPI_Abort" abort.out; then \
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

# Main mode (CharmMpiMain), for both mains: every process calls MPI_Allreduce
# from a nodegroup entry method, then the last PE calls CkExit(code). Every
# process must exit with code (exitstatus.sh prints each one's status; after
# a nonzero status mpirun may end the other processes' scripts before they
# print, so then at least one line is required; srun does not end them, so
# with a LAUNCH that starts with srun all NP lines are required), and CkExit
# must not return to its caller. A process that exits with status 0
# without MPI_Finalize makes mpirun fail the code 0 run.
testmain: mpi_main_exit mpi_main_exit_charmc
	@for prog in mpi_main_exit mpi_main_exit_charmc; do for code in 0 3; do \
	  $(TIMEOUT) $(LAUNCH) ./exitstatus.sh ./$$prog $$code mpi +pe $(PES) > main.out 2>&1; rc=$$?; \
	  cat main.out; \
	  if [ -n "$(TIMEOUT)" ] && [ $$rc -eq 124 ]; then \
	    echo "testmain: $$prog with code $$code did not end within the time limit ($(TIMEOUT))"; exit 1; fi; \
	  if [ $$rc -ne $$code ]; then \
	    echo "testmain: $$prog: launcher exit status $$rc, expected $$code"; exit 1; fi; \
	  n=`grep -c "^process exit status $$code$$" main.out`; \
	  all=`grep -c "^process exit status" main.out`; \
	  strict=no; case "$(LAUNCH)" in srun*) strict=yes;; esac; \
	  if [ $$n -ne $$all ] || [ $$n -eq 0 ] || { { [ $$code -eq 0 ] || [ $$strict = yes ]; } && [ $$n -ne $(NP) ]; }; then \
	    echo "testmain: $$prog: $$n of $$all reported processes ($(NP) started) exited with status $$code"; exit 1; fi; \
	  if ! grep -q "exitcode: MPI_Allreduce sum $(NP) correct" main.out; then \
	    echo "testmain: $$prog: no MPI_Allreduce result"; exit 1; fi; \
	  if grep -q "CkExit returned to its caller" main.out; then \
	    echo "testmain: $$prog: CkExit returned to its caller"; exit 1; fi; \
	  if grep -q "Warning: CharmMpiMain" main.out; then \
	    echo "testmain: $$prog: MPI_Finalize was not called on the main thread"; exit 1; fi; \
	done; done; \
	echo "mpi_main_exit testmain passed"

# Main mode: a CkAbort on the last PE must end the whole job through MPI_Abort.
testmainabort: mpi_main_exit
	@$(TIMEOUT) $(LAUNCH) ./mpi_main_exit 0 abort +pe $(PES) > mainabort.out 2>&1; rc=$$?; \
	cat mainabort.out; \
	if [ -n "$(TIMEOUT)" ] && [ $$rc -eq 124 ]; then \
	  echo "testmainabort: the run did not end within the time limit ($(TIMEOUT))"; exit 1; fi; \
	if [ $$rc -eq 0 ]; then echo "testmainabort: expected a nonzero exit"; exit 1; fi; \
	if ! grep -q "exitcode: deliberate abort" mainabort.out; then \
	  echo "testmainabort: the abort did not happen"; exit 1; fi; \
	if ! grep -q "MPI interop: ending the MPI job with MPI_Abort" mainabort.out; then \
	  echo "testmainabort: the job did not end through CharmMpiMain's abort hook"; exit 1; fi; \
	echo "mpi_main_exit testmainabort passed"

clean:
	rm -f mpi_main_exit mpi_main_exit_charmc main.out mainabort.out mpimain*.o
	rm -f mpi_interop mpi_exitcode *.o *.a *.decl.h *.def.h charmrun charm_all_libs.sh abort.out exit.out
