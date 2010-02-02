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
### End User configurable options ###

SHELL = /bin/sh
prefix     = 
top_srcdir = ..
srcdir     = ../env
VPATH=.:$(srcdir)

PROFLIB =
CFLAGS  =  -DHAVE_STDLIB_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STRING_H=1 -DUSE_STDARG=1 -DHAVE_LONG_DOUBLE=1 -DHAVE_LONG_LONG_INT=1 -DHAVE_PROTOTYPES=1 -DHAVE_SIGNAL_H=1 -DHAVE_SIGACTION=1 -DHAVE_SLEEP=1 -DHAVE_SYSCONF=1 $(OPTFLAGS) 
CCFLAGS = $(CFLAGS)
FFLAGS = $(OPTFLAGS) 
# Use LIBS to add any special libraries for C programs
LIBS =  
# Use FLIBS to add any special libraries for Fortran programs
FLIBS =  
EXECS = timers baseattr errhand sigchk timertest init gtime aborttest testerr \
	cmdline errhand2 getproc
OTHEREXECS = baseattrf errstringsf getprocf errhandf
default: $(EXECS)

#
# Note that runtests builds the executables as required
testing:
	-./runtests $(TESTARGS)

linktest: default clean

all: testing

init: init.o
	$(CLINKER) $(OPTFLAGS) -o init init.o $(LIBS)
timers: timers.o test.o
	$(CLINKER) $(OPTFLAGS) -o timers timers.o test.o $(LIBS)
timertest: timertest.o test.o
	$(CLINKER) $(OPTFLAGS) -o timertest timertest.o test.o $(LIBS)
baseattr: baseattr.o test.o
	$(CLINKER) $(OPTFLAGS) -o baseattr baseattr.o test.o $(LIBS)
gtime: gtime.o test.o
	$(CLINKER) $(OPTFLAGS) -o ./gtime gtime.o test.o $(LIBS) -lm

baseattrf: baseattrf.o
	$(FLINKER) $(OPTFLAGS) -o baseattrf baseattrf.o $(LIBS)

getprocf: getprocf.o
	$(FLINKER) $(OPTFLAGS) -o getprocf getprocf.o $(LIBS)
errhandf: errhandf.o
	$(FLINKER) $(OPTFLAGS) -o errhandf errhandf.o $(LIBS)

errstringsf: errstringsf.o
	$(FLINKER) $(OPTFLAGS) -o errstringsf errstringsf.o $(LIBS)

errhand: errhand.o test.o
	$(CLINKER) $(OPTFLAGS) -o errhand errhand.o test.o $(LIBS)
sigchk: sigchk.o test.o
	$(CLINKER) $(OPTFLAGS) -o sigchk sigchk.o test.o $(LIBS)
aborttest: aborttest.o
	$(CLINKER) $(OPTFLAGS) -o aborttest aborttest.o $(LIBS)
testerr: testerr.o
	$(CLINKER) $(OPTFLAGS) -o testerr testerr.o $(LIBS)
getproc: getproc.o
	$(CLINKER) $(OPTFLAGS) -o getproc getproc.o $(LIBS)
clean:
	@-rm -f *.o *~ PI* $(EXECS) core *.out *.orig env.diff \
	argstest $(OTHEREXECS) mpif.h ${srcdir}/*.o *.stdo
	@-rm -f work.pc work.pcl

.c:
	$(CC) $(CFLAGS) -c $<
	$(CLINKER) $(OPTFLAGS) -o $* $*.o $(LIBS)
.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
