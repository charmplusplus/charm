-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: comm_create_group

comm_create_group: comm_create_group.o
	$(CHARMC) -o comm_create_group comm_create_group.o

comm_create_group.o: comm_create_group.C
	$(CHARMC) -c comm_create_group.C

test: comm_create_group
	$(call run, ./comm_create_group +p1 +vp4)
	$(call run, ./comm_create_group +p2 +vp4)
	$(call run, ./comm_create_group +p2 +vp7)

testp: comm_create_group
	$(call run, ./comm_create_group +p$(P) +vp$(P) )

clean:
	rm -f *.o *.mod comm_create_group *~ conv-host charmrun comm_create_group.o charmrun.exe comm_create_group.exe comm_create_group.pdb comm_create_group.ilk ampirun
