-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all:	reg_pingpong

OBJS = pingpong.o

reg_pingpong: $(OBJS)
	$(CHARMC) -language charm++ -o reg_pingpong $(OBJS)

cifiles: pingpong.ci
	$(CHARMC)  pingpong.ci
	touch cifiles

pingpong.o: pingpong.C cifiles
	$(CHARMC) -c pingpong.C

test: all
	$(call run, +p1 ./reg_pingpong 1024 8192 2 1)
	$(call run, +p2 ./reg_pingpong 1024 8192 2 1)

test-bench: all
	$(call run, ./reg_pingpong +p1)
	$(call run, ./reg_pingpong +p2)

clean:
	rm -f *.decl.h *.def.h conv-host *.o reg_pingpong charmrun cifiles
