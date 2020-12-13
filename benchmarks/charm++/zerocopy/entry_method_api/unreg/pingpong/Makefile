-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all:	unreg_pingpong

OBJS = pingpong.o

unreg_pingpong: $(OBJS)
	$(CHARMC) -language charm++ -o unreg_pingpong $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

pingpong.o: pingpong.C cifiles
	$(CHARMC) -c pingpong.C

test: all
	$(call run, +p1 ./unreg_pingpong 1024 8192 2 1)
	$(call run, +p2 ./unreg_pingpong 1024 8192 2 1)

test-bench: all
	$(call run, ./unreg_pingpong +p1)
	$(call run, ./unreg_pingpong +p2)

clean:
	rm -f *.decl.h *.def.h conv-host *.o unreg_pingpong charmrun cifiles
