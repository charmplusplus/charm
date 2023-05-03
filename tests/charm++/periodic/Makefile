-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: periodic

periodic: periodic.decl.h periodic.def.h periodic.C
	$(CHARMC) -language charm++ periodic.C -o periodic

periodic.decl.h periodic.def.h: periodic.ci
	$(CHARMC) periodic.ci

clean:
	rm -f *.decl.h *.def.h *.o periodic charmrun

test: all
	$(call run, ./periodic +p1 )
	$(call run, ./periodic +p2 )

testp: all
	$(call run, ./periodic +p$(P))

smptest:
	$(call run, ./periodic +p2 ++ppn 2)
	$(call run, ./periodic +p4 ++ppn 2)
