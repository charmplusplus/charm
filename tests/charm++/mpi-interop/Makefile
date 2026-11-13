-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)
MPICXX ?= mpicxx
# LAUNCH must be the launcher the MPI library requires, starting NP
# processes, e.g. LAUNCH="srun --mpi=cray_shasta -n 2" for Cray MPICH.
NP ?= 2
LAUNCH ?= mpirun -np $(NP)
PES ?= 2

all: mpi_interop

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

test: all
	$(LAUNCH) ./mpi_interop $(NP) +pe $(PES)
	@$(MAKE) --no-print-directory testabort

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

clean:
	rm -f mpi_interop *.o *.a *.decl.h *.def.h charmrun charm_all_libs.sh abort.out
