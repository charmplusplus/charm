# Generated automatically from Makefile.in by configure.
ALL: default
##### User configurable options #####

MPIR_HOME    = 
CC           = ../../../../bin/ampiCC
CLINKER      = ../../../../bin/ampiCC
CCC          = 
CCLINKER     = $(CCC)
F77          = :
FLINKER      = :
OPTFLAGS     = 
MPIFDEP      = 
### End User configurable options ###

# Add .f90 as a known suffix
.SUFFIXES: .f90

SHELL = /bin/sh
prefix     = 
top_srcdir = ..
srcdir     = ../topol
VPATH=.:$(srcdir)

PROFLIB =
CFLAGS  =  -DHAVE_STDLIB_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STRING_H=1 -DUSE_STDARG=1 -DHAVE_LONG_DOUBLE=1 -DHAVE_LONG_LONG_INT=1 -DHAVE_PROTOTYPES=1 -DHAVE_SIGNAL_H=1 -DHAVE_SIGACTION=1 -DHAVE_SLEEP=1 -DHAVE_SYSCONF=1 $(OPTFLAGS) 
CCFLAGS = $(CFLAGS)
FFLAGS = $(OPTFLAGS) 
# Use LIBS to add any special libraries for C programs
LIBS =  
# Use FLIBS to add any special libraries for Fortran programs
FLIBS =  
EXECS = cart cart2 graphtest cartmap cartorder dims

default: $(EXECS)

#
# Note that runtests builds the executables as required
testing: 
	-./runtests $(TESTARGS)

linktest: default clean

all: testing

fortran_tests: cartf twod cart1f cart2f

cart: cart.o test.o
	$(CLINKER) $(OPTFLAGS) -o cart cart.o test.o $(LIBS)

cartmap: cartmap.o
	$(CLINKER) $(OPTFLAGS) -o cartmap cartmap.o $(LIBS)

cartorder: cartorder.o test.o
	$(CLINKER) $(OPTFLAGS) -o cartorder cartorder.o test.o $(LIBS)

cart2: cart2.o test.o
	$(CLINKER) $(OPTFLAGS) -o cart2 cart2.o test.o $(LIBS)

graphtest: graphtest.o
	$(CLINKER) $(OPTFLAGS) -o graphtest graphtest.o $(LIBS)

cartf: $(MPIFDEP) cartf.o
	$(FLINKER) $(OPTFLAGS) -o cartf cartf.o $(FLIBS)

cart1f: $(MPIFDEP) cart1f.o
	$(FLINKER) $(OPTFLAGS) -o cart1f cart1f.o $(FLIBS)

cart2f: $(MPIFDEP) cart2f.o
	$(FLINKER) $(OPTFLAGS) -o cart2f cart2f.o $(FLIBS)

twod: $(MPIFDEP) twod.o
	$(FLINKER) $(OPTFLAGS) -o twod twod.o $(FLIBS)

clean:
	@-rm -f *.o *~ core PI* $(EXECS) cartf twod cart1f cart2f \
	*.out topol.diff *.trace mpif.h ${srcdir}/*.o *.stdo
	@-rm -f work.pc work.pcl

.c:
	$(CLINKER) $(OPTFLAGS) -o $* $< $(LIBS)
.c.o:
	$(CC) $(CFLAGS) -c $<
.f:
	$(FLINKER) $(OPTFLAGS) -o $* $< $(FLIBS)
.f.o:
	$(F77) $(FFLAGS) -c $<

# The following allows us to build a Fortran 90 test with those compilers that 
# allow .f90 extensions (cartc.f90)
.f90:
	$(F90) $(F90FLAGS) -o $* $< $(FLIBS)
