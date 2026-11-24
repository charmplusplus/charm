-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: intercomm_merge

intercomm_merge: intercomm_merge.o
	$(CHARMC) -o intercomm_merge intercomm_merge.o

intercomm_merge.o: intercomm_merge.C
	$(CHARMC) -c intercomm_merge.C

test: intercomm_merge
	$(call run, ./intercomm_merge +p1 +vp4)
	$(call run, ./intercomm_merge +p2 +vp4)
	$(call run, ./intercomm_merge +p1 +vp8)

testp: intercomm_merge
	$(call run, ./intercomm_merge +p$(P) +vp$(P) )
	$(call run, ./intercomm_merge +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod intercomm_merge *~ conv-host charmrun intercomm_merge.o charmrun.exe intercomm_merge.exe intercomm_merge.pdb intercomm_merge.ilk ampirun
