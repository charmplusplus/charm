-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: collmatch

collmatch: collmatch.o
	$(CHARMC) -o collmatch collmatch.o

collmatch.o: collmatch.C
	$(CHARMC) -c collmatch.C

# Part 8 of the test keeps a non-blocking collective outstanding across
# AMPI_Migrate. In AMPI's default migration mode that makes AMPI_Migrate abort
# (charm #4072 item 8, #4093); +ampiMigrateWithCollectives allows it, so every
# run carries the flag. The +balancer RandCentLB runs move every rank while
# the collective is outstanding; they were kept out of `make test` by #4091
# until #4093 made the broadcast delivery to migrating ranks reliable.
test: collmatch
	$(call run, ./collmatch +p1 +vp4 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p1 +vp7 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp4 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp7 +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp4 +balancer RandCentLB +ampiMigrateWithCollectives)
	$(call run, ./collmatch +p2 +vp7 +balancer RandCentLB +ampiMigrateWithCollectives)

testp: collmatch
	$(call run, ./collmatch +p$(P) +vp$(P) )
	$(call run, ./collmatch +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod collmatch *~ conv-host charmrun ampirun collmatch.exe collmatch.pdb collmatch.ilk
