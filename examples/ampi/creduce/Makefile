-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: creduce

ampi: clean creduce

creduce: test.C
	$(CHARMC) -c test.C
	$(CHARMC) -o creduce test.o

mpi: test.C
	mpiCC -o creduce test.C $(OPTS)

clean:
	rm -f *.o *.mod creduce *~ conv-host charmrun ampirun

test: creduce
	$(call run, ./creduce +p1 +vp1 )
	$(call run, ./creduce +p1 +vp2 )
	$(call run, ./creduce +p1 +vp4 )
	$(call run, ./creduce +p2 +vp1 )
	$(call run, ./creduce +p2 +vp2 )
	$(call run, ./creduce +p2 +vp4 )

testp: creduce
	$(call run, ./creduce +p$(P) +vp$(P) )
	$(call run, ./creduce +p$(P) +vp$$(( $(P) * 2 )) )
	$(call run, ./creduce +p$(P) +vp$$(( $(P) * 4 )) )
