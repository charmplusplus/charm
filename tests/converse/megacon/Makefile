-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = \
  blkinhand.o \
  megacon.o \
  ringsimple.o \
  ring.o \
  fibobj.o \
  fibthr.o \
  broadc.o \
  priotest.o \
  deadlock.o \
  vars.o \
  nodenum.o \
  specmsg.o \
  bigmsg.o \
  vecsend.o \
  posixth.o \
  future.o \
  multicast.o \
  multisend.o \
  handler.o \
  reduction.o \
  nodereduction.o \

all: megacon

megacon: $(OBJS)
	$(CHARMC) -o megacon $(OBJS) -language converse++

megacon.o: megacon.c
	$(CHARMC) -gen-cpm megacon.c
	$(CHARMC) megacon.c

blkinhand.o: blkinhand.c
	$(CHARMC) -gen-cpm blkinhand.c
	$(CHARMC) blkinhand.c

ring.o: ring.c
	$(CHARMC) -gen-cpm ring.c
	$(CHARMC) ring.c

ringsimple.o: ringsimple.c
	$(CHARMC) ringsimple.c

vecsend.o: vecsend.c
	$(CHARMC) -gen-cpm vecsend.c
	$(CHARMC) vecsend.c

multicast.o: multicast.c
	$(CHARMC) multicast.c

fibobj.o: fibobj.c
	$(CHARMC) -gen-cpm fibobj.c
	$(CHARMC) fibobj.c

fibthr.o: fibthr.c
	$(CHARMC) -gen-cpm fibthr.c
	$(CHARMC) fibthr.c

broadc.o: broadc.c
	$(CHARMC) broadc.c

priotest.o: priotest.c
	$(CHARMC) -gen-cpm priotest.c
	$(CHARMC) priotest.c

deadlock.o: deadlock.c
	$(CHARMC) deadlock.c

vars.o: vars.c
	$(CHARMC) -gen-cpm vars.c
	$(CHARMC) vars.c

nodenum.o: nodenum.c
	$(CHARMC) -gen-cpm nodenum.c
	$(CHARMC) nodenum.c

future.o: future.c
	$(CHARMC) -gen-cpm future.c
	$(CHARMC) future.c

specmsg.o: specmsg.c
	$(CHARMC) -gen-cpm specmsg.c
	$(CHARMC) specmsg.c

bigmsg.o: bigmsg.c
	$(CHARMC) -gen-cpm bigmsg.c
	$(CHARMC) bigmsg.c

posixth.o: posixth.c
	$(CHARMC) -gen-cpm posixth.c
	$(CHARMC) posixth.c

multisend.o: multisend.c
	$(CHARMC) multisend.c

handler.o: handler.c
	$(CHARMC) handler.c

reduction.o: reduction.c
	$(CHARMC) reduction.c

nodereduction.o: nodereduction.c
	$(CHARMC) nodereduction.c

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f megacon
	rm -f conv-host charmrun charmrun.exe megacon.exe megacon.pdb megacon.ilk

test: megacon
	$(call run, ./megacon +p1 )
	$(call run, ./megacon +p2 )
	$(call run, ./megacon +p3 )
	$(call run, ./megacon +p4 )

smptest: megacon
	$(call run, ./megacon +p2 ++ppn 2)
	$(call run, ./megacon +p4 ++ppn 2)

testp: megacon
	$(call run, ./megacon +p$(P) )
