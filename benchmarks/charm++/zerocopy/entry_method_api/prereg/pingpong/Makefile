-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all:	prereg_pingpong

OBJS = pingpong.o

prereg_pingpong: $(OBJS)
	$(CHARMC) -language charm++ -o prereg_pingpong $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

pingpong.o: pingpong.C cifiles
	$(CHARMC) -c pingpong.C

test: all
	$(call run, +p1 ./prereg_pingpong 1024 8192 2 1)
	$(call run, +p2 ./prereg_pingpong 1024 8192 2 1)

test-bench: all
	$(call run, ./prereg_pingpong +p1)
	$(call run, ./prereg_pingpong +p2)

clean:
	rm -f *.decl.h *.def.h conv-host *.o prereg_pingpong charmrun cifiles
