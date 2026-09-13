-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = osu_latency.o

all: osu_latency

osu_latency: $(OBJS)
	$(CHARMC) -language charm++ -o osu_latency $(OBJS)

osu_latency.decl.h: osu_latency.ci
	$(CHARMC) osu_latency.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o osu_latency charmrun

osu_latency.o: osu_latency.C osu_latency.decl.h
	$(CHARMC) -c osu_latency.C

test: all
	$(call run, +p2 ./osu_latency)
