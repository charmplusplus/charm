-include ../../common.mk
CHARMC=../../../bin/ampif90 $(OPTS)

all: pgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o

pgm.o: pgm.f90
	$(CHARMC) -c pgm.f90

#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun ampirun

test: pgm
	$(call run, ./pgm +p2 +vp4 )
