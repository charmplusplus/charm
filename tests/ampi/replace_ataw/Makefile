-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: replace_ataw

replace_ataw: replace_ataw.o
	$(CHARMC) -o replace_ataw replace_ataw.o

replace_ataw.o: replace_ataw.C
	$(CHARMC) -c replace_ataw.C

test: replace_ataw
	$(call run, ./replace_ataw +p1 +vp4)
	$(call run, ./replace_ataw +p2 +vp4)
	$(call run, ./replace_ataw +p2 +vp5)

testp: replace_ataw
	$(call run, ./replace_ataw +p$(P) +vp$(P) )

clean:
	rm -f *.o *.mod replace_ataw *~ conv-host charmrun replace_ataw.o charmrun.exe replace_ataw.exe replace_ataw.pdb replace_ataw.ilk ampirun
