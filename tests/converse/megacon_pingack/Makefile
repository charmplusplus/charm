-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = \
  megacon.o \
  bigmsg.o \

all: megacon

megacon: $(OBJS)
	$(CHARMC) -o megacon $(OBJS) -language converse++

megacon.o: megacon.c
	$(CHARMC) -gen-cpm megacon.c
	$(CHARMC) megacon.c

bigmsg.o: bigmsg.c
	$(CHARMC) -gen-cpm bigmsg.c
	$(CHARMC) bigmsg.c

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f megacon
	rm -f conv-host charmrun charmrun.exe megacon.exe megacon.pdb megacon.ilk

test: megacon
	$(call run, ./megacon +p2 )

smptest: megacon
	$(call run, ./megacon +p2 ++ppn 2)
	$(call run, ./megacon +p4 ++ppn 2)

testp: megacon
	$(call run, ./megacon +p$(P) )
