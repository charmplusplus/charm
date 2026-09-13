-include ../../common.mk
CHARMC=../../../bin/charmc -O3 $(OPTS)

OBJS = osu_bw.o

all: osu_bw

osu_bw: $(OBJS)
	$(CHARMC) -language charm++ -o osu_bw $(OBJS)

proj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o osu_bw.prj $(OBJS)

osu_bw.o: osu_bw.C osu_bw.decl.h
	$(CHARMC) -c osu_bw.C

osu_bw.decl.h: osu_bw.ci
	$(CHARMC)  osu_bw.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o osu_bw osu_bw.prj charmrun *~ *log *projrc *sts

test: all
	$(call run, +p4 ./osu_bw 32 32 4 )

testp: all
	$(call run, +p$(P) ./osu_bw $$(( $(P) * 80 ))  $$(( $(P) * 80 )) $$(( $(P) * 10 )) )
