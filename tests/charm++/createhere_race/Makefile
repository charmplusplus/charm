-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = createhere_race.o

all: createhere_race

createhere_race: $(OBJS)
	$(CHARMC) -language charm++ -o createhere_race $(OBJS)

createhere_race.decl.h: createhere_race.ci
	$(CHARMC) createhere_race.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o createhere_race charmrun *.log *.sum *.sts

createhere_race.o: createhere_race.C createhere_race.decl.h
	$(CHARMC) -c createhere_race.C

test: all
	$(call run, ./createhere_race +p4 )

testp: all
	$(call run, ./createhere_race +p$(P) )

smptest: all
	$(call run, ./createhere_race +p2 ++ppn 2)
	$(call run, ./createhere_race +p4 ++ppn 2)
