# Generated automatically from Makefile.in by configure.
ALL: default
##### User configurable options #####

MPIR_HOME   = 
CC          = ../../../../bin/ampiCC
CLINKER     = ../../../../bin/ampiCC
CCC         = 
CCLINKER    = $(CCC)
F77         = :
FLINKER     = :
OPTFLAGS    = 

### End User configurable options ###

SHELL = /bin/sh
prefix     = 
top_srcdir = ..
srcdir     = ../profile
VPATH=.:$(srcdir)

PROFLIB =
CFLAGS  =  -DHAVE_STDLIB_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STRING_H=1 -DUSE_STDARG=1 -DHAVE_LONG_DOUBLE=1 -DHAVE_LONG_LONG_INT=1 -DHAVE_PROTOTYPES=1 -DHAVE_SIGNAL_H=1 -DHAVE_SIGACTION=1 -DHAVE_SLEEP=1 -DHAVE_SYSCONF=1 $(OPTFLAGS) 
CCFLAGS = $(CFLAGS)
FFLAGS = $(OPTFLAGS) 
# Use LIBS to add any special libraries for C programs
LIBS =  
# Use FLIBS to add any special libraries for Fortran programs
FLIBS =  
EXECS = ptest colluses
OTHEREXECS = 
default: $(EXECS)

#
# Note that runtests builds the executables as required
testing: 
	  -./runtests $(TESTARGS)

all: testing

fortran_tests: 

ptest: ptest.o 
	$(CLINKER) $(OPTFLAGS) -o ptest ptest.o $(LIBS)

colluses: colluses.o
	$(CLINKER) $(OPTFLAGS) -o colluses colluses.o $(LIBS)

#
# Make sure that we remove executables for specific architectures
clean:
	@-rm -f *.o *~ PI* $(EXECS) *.out core profile.diff \
		$(OTHEREXECS) mpif.h ${srcdir}/*.o *.stdo
	@-for file in $(EXECS) Makefile ; do \
	   rm -f $$file.sun4 $$file.alpha $$file.IRIX $$file.freebsd ;\
	done

.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
