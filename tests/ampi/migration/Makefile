-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: migration

migration: test.o
	$(CHARMC) -pieglobals -o migration test.o

test.o: test.C
	$(CHARMC) -pieglobals -c test.C

#
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod migration *~ conv-host charmrun test.o charmrun.exe migration.exe migration.pdb migration.ilk ampirun

test: migration
	$(call run, ./migration +p1 +vp1 +ampiEnableMigrateToPe )
	$(call run, ./migration +p1 +vp4 +ampiEnableMigrateToPe )
	$(call run, ./migration +p1 +vp20 +ampiEnableMigrateToPe )
	$(call run, ./migration +p2 +vp1 +ampiEnableMigrateToPe )
	$(call run, ./migration +p2 +vp4 +ampiEnableMigrateToPe --test_mapfile +mapping MAPFILE )
	$(call run, ./migration +p2 +vp20 +ampiEnableMigrateToPe --test_mapfile +mapping MAPFILE )

testp: migration
	$(call run, ./migration +p$(P) +vp$(P) +ampiEnableMigrateToPe )
	$(call run, ./migration +p$(P) +vp$$(( $(P) * 2 )) +ampiEnableMigrateToPe )
	$(call run, ./migration +p$(P) +vp$$(( $(P) * 4 )) +ampiEnableMigrateToPe )
