-include ../../common.mk
CHARMC=../../../bin/ampif90 $(OPTS)

all: fallreduce

fallreduce: pgm.o
	$(CHARMC) -o fallreduce pgm.o

pgm.o: pgm.f90
	$(CHARMC) -c pgm.f90

#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod fallreduce *~ conv-host charmrun ampirun

test: fallreduce
	$(call run, ./fallreduce +p2 +vp4 )

testp: fallreduce
	$(call run, ./fallreduce +p$(P) +vp$(P))
	$(call run, ./fallreduce +p$(P) +vp$$(( $(P) * 2 )))
	$(call run, ./fallreduce +p$(P) +vp$$(( $(P) * 4 )))
