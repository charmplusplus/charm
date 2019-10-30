-include ../../../../../common.mk
-include ../../../../../../include/conv-mach-opt.mak
CHARMC=../../../../../../bin/charmc $(OPTS)

all: get_put_pingpong

OBJS = get_put_pingpong.o

get_put_pingpong: $(OBJS)
	$(CHARMC) -language charm++ -o get_put_pingpong $(OBJS)

cifiles: get_put_pingpong.ci
	$(CHARMC)  get_put_pingpong.ci
	touch cifiles

get_put_pingpong.o: get_put_pingpong.C cifiles
	$(CHARMC) -c get_put_pingpong.C

test: all
	$(call run, +p1 ./get_put_pingpong 20 )
	$(call run, +p2 ./get_put_pingpong 20 )
ifeq ($(CMK_USE_CMA),1)
	$(call run, +p2 ./get_put_pingpong 20 +noCMAForZC )
endif

clean:
	rm -f *.decl.h *.def.h conv-host *.o get_put_pingpong charmrun cifiles
