-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: pingpong

OBJS = pingpong.o

pingpong: $(OBJS)
	$(CHARMC) -language charm++ -o pingpong $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

pingpong.o: pingpong.C cifiles
	$(CHARMC) -c pingpong.C

test: all
	$(call run, +p1 ./pingpong 1024 8192 2 1)
	$(call run, +p2 ./pingpong 1024 8192 2 1)

test-bench: all
	$(call run, +p1 ./pingpong)
	$(call run, +p2 ./pingpong)

clean:
	rm -f *.decl.h *.def.h conv-host *.o pingpong charmrun cifiles
