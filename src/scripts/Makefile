#############################################################################"
#
# This one Makefile serves all of converse and charm.  It includes
# rules for building:
#
#     * The CONVERSE libraries
#     * Charm++ runtime
#     * ITC++ Translator
#     * Charm++ IDL Translator
#     * Languages
# 
# This Makefile is common to all machines.  This works because the charmc
# is smart enough to "conceal" the differences between machines.  charmc
# gets its smarts by reading the conv-mach.sh configuration file.
#
# If you wish to compile with debugging, optimization, or some other charm
# option, you can do so by setting OPTS.  For example, if you need a lot
# of debugging information, you could type:
#
#     make charm++ OPTS="-g -save -verbose"
#
###############################################################################

SHELL=/bin/sh

all: charm++ libs IDL tsm sm pvm sdag

charm++: converse charmxi charmlibs

IDL: charm++ charmidl libidl.a

tsm: converse libtsm.a

sm: converse libsm.a

pvm: converse libpvmc.a

sdag: charm++ sdagx libsdag.a

translators: sdagx charmxi conv-cpm charmidl

libs: convlibs charmlibs
	cd libs; make otherlibs OPTS='$(OPTS)'

test-all: all
	cd ../pgms ; make test-all  OPTS='$(OPTS)'

test-converse: converse
	cd ../pgms ; make test-converse  OPTS='$(OPTS)'

CHARMC=../bin/charmc $(CHARMOPTS) $(OPTS)

SEQCHARMC=../bin/charmc -seq $(CHARMOPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.sh)
#
###############################################################################

ALLHEADERS=charm++.h ckstream.h charm.h cpthreads.h converse.h \
	conv-trace.h conv-mach.h simplemsg.h sm.h trace.h pvmc.h \
	pvm3.h sdag.h CDep.h CCounter.h CMsgBuffer.h CWhenTrigger.h \
	TList.h idl.h ckarray.h tempo.h waitqd.h LBDatabase.h \
	lbdb.h LBDBManager.h LBComm.h LBOM.h LBObj.h LBMachineUtil.h \
	Refiner.h ObjGraph.h CentralLB.h RandCentLB.h \
	RecBisectBfLB.h graph.h fifoInt.h bitvecset.h \
	RefineLB.h MetisLB.h  Comm1LB.h \
	HeapCentLB.h Set.h elements.h heap.h conv-ccs.h NeighborLB.h \
	WSLB.h CkLists.h GreedyRefLB.h RandRefLB.h CommLB.h CommLBHeap.h

ALLINTERFACES=idl.ci

basics: dirs+sources $(ALLHEADERS) $(ALLINTERFACES) charmc conv-mach.sh
	rm -rf ../include
	mkdir ../include
	cp $(ALLHEADERS) ../include/
	cp $(ALLINTERFACES) ../include/
	rm -f ../bin/charmc ; cp charmc ../bin/
	chmod 755 ../bin/charmc
	cp conv-mach.sh ../include/
	touch basics

dirs+sources:
	@rm -f .vdir .gdir
	@(cd .. ; basename `pwd`) 			> .vdir
	@(cd .. ; basename `pwd` | sed -e 's@-.*@@') 	> .gdir
	@echo
	@echo 'Gathering source code.'
	@echo
	find . -type l -exec rm {} \;
	rm -rf QuickThreads
	rm -rf ../bin ; mkdir ../bin
	rm -rf ../lib ; mkdir ../lib
	rm -rf ../src ; mkdir ../src
	rm -rf ../pgms
	rm -rf ../libsrc
	../../src/Common/scripts/gatherflat ../../src/Common/scripts .
	./gatherflat ../../src/Common/conv-core        .
	./gatherflat ../../src/Common/conv-ldb         .
	./gatherflat ../../src/Common/ck-core          .
	./gatherflat ../../src/Common/ck-perf          .
	./gatherflat ../../src/Common/ck-ldb           .
	./gatherflat ../../src/Common/langs/simplemsg  .
	./gatherflat ../../src/Common/langs/pvmc       .
	./gatherflat ../../src/Common/langs/sdag/xlator       .
	./gatherflat ../../src/Common/langs/sdag/runtime      .
	./gatherflat ../../src/Common/xlat-i           .
	./gatherflat ../../src/Common/xlat-idl         .
	./gatherflat ../../src/Common/xlatcpm          .
	./gathertree ../../src/QuickThreads QuickThreads
	./gathertree ../../src/Common/libs libs
	./gatherflat ../../src/Common.`cat .gdir`      .
	./gatherflat ../../src/`cat .vdir`             .
	./gathertree ../../src/Common/langs   ../src/langs
	./gathertree ../../src/Common/xlat-i  ../src/xlat-i
	./gathertree ../../src/Common/xlat-idl  ../src/xlat-idl
	./gathertree ../../src/Common/xlatcpm ../src/xlatcpm
	./gathertree ../../pgms ../pgms
	./gathertree ../../src/Common/libs ../libsrc
	ln -s ../../src/Common/scripts/Makefile ../src
	@echo
	@echo
	@rm -f .vdir .gdir
	touch dirs+sources


###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h conv-mach.sh \
          ../include/CkFutures.decl.h CkFutures.def.h \
          ../include/CkArray.decl.h CkArray.def.h \
          ../include/LBDatabase.decl.h LBDatabase.def.h \
          ../include/CentralLB.decl.h CentralLB.def.h \
          ../include/RandCentLB.decl.h RandCentLB.def.h \
          ../include/RecBisectBfLB.decl.h RecBisectBfLB.def.h \
          ../include/MetisLB.decl.h MetisLB.def.h \
          ../include/RefineLB.decl.h RefineLB.def.h \
          ../include/HeapCentLB.decl.h HeapCentLB.def.h \
          ../include/NeighborLB.decl.h NeighborLB.def.h \
          ../include/WSLB.decl.h WSLB.def.h \
          ../include/GreedyRefLB.decl.h GreedyRefLB.def.h \
          ../include/RandRefLB.decl.h RandRefLB.def.h \
          ../include/tempo.decl.h tempo.def.h \
          ../include/waitqd.decl.h waitqd.def.h \
	  ../include/CommLB.decl.h CommLB.def.h  \
	  ../include/Comm1LB.decl.h Comm1LB.def.h 

TRACELIBS=libtrace-none.a libtrace-projections.a libtrace-summary.a

CVLIBS=libconv-core.a libconv-cplus-n.a libconv-cplus-y.a libldb-rand.o libldb-spray.o libldb-graph.o libldb-test.o $(TRACELIBS)

LIBCONV_CORE=threads.o convcore.o conv-conds.o spantree.o queueing.o fifo.o msgmgr.o memory.o cpm.o cpthreads.o futures.o cldb.o random.o debug-message.o debug-object.o info.o generate.o edgelist.o conv-ccs.o

converse: basics QuickThreads/libckqt.a $(CVLIBS) conv-host-if-needed conv-cpm convlibs

conv-host-if-needed:
	if [ -f conv-host.c ] ; then make conv-host OPTS='$(OPTS)' ; fi

QuickThreads/libckqt.a:
	. ./conv-mach.sh;cd QuickThreads;./configure $$CMK_QT
	cd QuickThreads ; make qt  OPTS='$(OPTS)'
	cp QuickThreads/libqt.a ../lib/libckqt.a

conv-host: conv-host.c
	../bin/charmc $(OPTS) -c -seq -DSILENT conv-host.c
	../bin/charmc $(OPTS) -cp ../bin -seq -o conv-host-silent conv-host.o
	../bin/charmc $(OPTS) -c -seq -DNOTIFY conv-host.c
	../bin/charmc $(OPTS) -cp ../bin -seq -o conv-host-notify conv-host.o
	cp conv-host-notify conv-host
	cp conv-host ../bin/conv-host

convlibs:
	cd libs; make convlibs OPTS='$(OPTS)'

libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCONV_CORE)

libconv-cplus-y.a: machine.c $(CVHEADERS)	
	$(CHARMC) -cp ../lib/ -o $@ -DFOR_CPLUS=1 machine.c

libconv-cplus-n.a: machine.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ -DFOR_CPLUS=0 machine.c

cpm.o: cpm.c $(CVHEADERS)
	$(CHARMC) cpm.c

fifo.o: fifo.c $(CVHEADERS)
	$(CHARMC) fifo.c

threads.o: threads.c $(CVHEADERS)
	$(CHARMC) -IQuickThreads threads.c

spantree.o: spantree.c $(CVHEADERS)
	$(CHARMC) spantree.c

convcore.o: convcore.c $(CVHEADERS)
	$(CHARMC) convcore.c

conv-ccs.o: conv-ccs.c $(CVHEADERS)
	$(CHARMC) conv-ccs.c

random.o: random.c $(CVHEADERS)
	$(CHARMC) random.c

queueing.o: queueing.c $(CVHEADERS)
	$(CHARMC) queueing.c

conv-conds.o: conv-conds.c $(CVHEADERS)
	$(CHARMC) conv-conds.c

futures.o: futures.c $(CVHEADERS)
	$(CHARMC) futures.c

msgmgr.o: msgmgr.c $(CVHEADERS)
	$(CHARMC) msgmgr.c

cpthreads.o: cpthreads.c $(CVHEADERS)
	$(CHARMC) cpthreads.c

memory.o: memory.c $(CVHEADERS)
	$(CHARMC) memory.c

edgelist.o : edgelist.c ../../src/Common/conv-ldb/typedefs.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ edgelist.c

generate.o : generate.c ../../src/Common/conv-ldb/typedefs.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ generate.c

cldb.o: cldb.c ../../src/Common/conv-ldb/cldb.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ cldb.c

libldb-rand.o: cldb.rand.c ../../src/Common/conv-ldb/cldb.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ -cp ../lib/ -o $@ cldb.rand.c

libldb-graph.o: cldb.graph.c ../../src/Common/conv-ldb/cldb.graph.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ -cp ../lib/ -o $@ cldb.graph.c 

libldb-spray.o: cldb.spray.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cldb.spray.c

libldb-test.o: cldb.test.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cldb.test.c

###############################################################################
#
# Charm Libraries
#
###############################################################################

CKHEADERS=ck.h ckstream.h envelope.h init.h qd.h charm.h charm++.h trace.h \
          ckfutures.h ckarray.h tempo.h waitqd.h LBDatabase.h lbdb.h \
	  LBDBManager.h LBComm.h LBOM.h LBObj.h LBMachineUtil.h \
	  Refiner.h ObjGraph.h heap.h elements.h CommLBHeap.h\
	  CentralLB.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h HeapCentLB.h CommLB.h Comm1LB.h\
	  MetisLB.h NeighborLB.h WSLB.h GreedyRefLB.h RandRefLB.h CkLists.h \
	  $(CVHEADERS)

CK_LIBS_CORE=libck.a

LIBCK_CORE=init.o register.o qd.o ck.o main.o msgalloc.o ckfutures.o \
           ckarray.o tempo.o waitqd.o LBDatabase.o lbdb.o \
	   LBDBManager.o LBComm.o LBObj.o LBMachineUtil.o Refiner.o \
	   ObjGraph.o \
	   CentralLB.o RandCentLB.o RecBisectBfLB.o graph.o bitvecset.o fifoInt.o \
	   MetisLB.o RefineLB.o Set.o CommLB.o Comm1LB.o\
           HeapCentLB.o heap.o NeighborLB.o WSLB.o GreedyRefLB.o \
	   RandRefLB.o CommLBHeap.o

charmlibs: converse $(CK_LIBS_CORE)
	cd libs; make charmlibs OPTS='$(OPTS)'

libck.a: $(LIBCK_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE)

LIBTRACE_NONE=trace-none.o
libtrace-none.a: $(LIBTRACE_NONE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_NONE)

LIBTRACE_PROJ=trace-projections.o
libtrace-projections.a: $(LIBTRACE_PROJ)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_PROJ)

LIBTRACE_SUMM=trace-summary.o
libtrace-summary.a: $(LIBTRACE_SUMM)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_SUMM)

../include/CkArray.decl.h : CkArray.decl.h
	/bin/cp CkArray.decl.h ../include

../include/LBDatabase.decl.h : LBDatabase.decl.h
	/bin/cp LBDatabase.decl.h ../include

../include/CentralLB.decl.h : CentralLB.decl.h
	/bin/cp CentralLB.decl.h ../include

../include/RandCentLB.decl.h : RandCentLB.decl.h
	/bin/cp RandCentLB.decl.h ../include

../include/RecBisectBfLB.decl.h : RecBisectBfLB.decl.h
	/bin/cp RecBisectBfLB.decl.h ../include

../include/MetisLB.decl.h : MetisLB.decl.h
	/bin/cp MetisLB.decl.h ../include

../include/RefineLB.decl.h : RefineLB.decl.h
	/bin/cp RefineLB.decl.h ../include

../include/CommLB.decl.h : CommLB.decl.h
	/bin/cp CommLB.decl.h ../include

../include/Comm1LB.decl.h : Comm1LB.decl.h
	/bin/cp Comm1LB.decl.h ../include

../include/HeapCentLB.decl.h : HeapCentLB.decl.h
	/bin/cp HeapCentLB.decl.h ../include

../include/NeighborLB.decl.h : NeighborLB.decl.h
	/bin/cp NeighborLB.decl.h ../include

../include/WSLB.decl.h : WSLB.decl.h
	/bin/cp WSLB.decl.h ../include

../include/GreedyRefLB.decl.h : GreedyRefLB.decl.h
	/bin/cp GreedyRefLB.decl.h ../include

../include/RandRefLB.decl.h : RandRefLB.decl.h
	/bin/cp RandRefLB.decl.h ../include

../include/CkFutures.decl.h : CkFutures.decl.h
	/bin/cp CkFutures.decl.h ../include

CkFutures.decl.h CkFutures.def.h : ckfutures.ci charmxi
	$(CHARMC) ckfutures.ci

../include/tempo.decl.h : tempo.decl.h
	/bin/cp tempo.decl.h ../include

tempo.decl.h tempo.def.h : tempo.ci charmxi
	$(CHARMC) tempo.ci

../include/waitqd.decl.h : waitqd.decl.h
	/bin/cp waitqd.decl.h ../include

waitqd.decl.h waitqd.def.h : waitqd.ci charmxi
	$(CHARMC) waitqd.ci

CkArray.decl.h CkArray.def.h : ckarray.ci charmxi
	$(CHARMC) ckarray.ci

LBDatabase.decl.h LBDatabase.def.h : LBDatabase.ci charmxi
	$(CHARMC) LBDatabase.ci

CentralLB.decl.h Central.def.h : CentralLB.ci charmxi
	$(CHARMC) CentralLB.ci

RandCentLB.decl.h RandCent.def.h : RandCentLB.ci charmxi
	$(CHARMC) RandCentLB.ci

RecBisectBfLB.decl.h RandCent.def.h : RecBisectBfLB.ci charmxi
	$(CHARMC) RecBisectBfLB.ci

MetisLB.decl.h MetisLB.def.h : MetisLB.ci charmxi
	$(CHARMC) MetisLB.ci

RefineLB.decl.h RefineLB.def.h : RefineLB.ci charmxi
	$(CHARMC) RefineLB.ci

CommLB.decl.h CommLB.def.h : CommLB.ci charmxi
	$(CHARMC) CommLB.ci

Comm1LB.decl.h Comm1LB.def.h : Comm1LB.ci charmxi
	$(CHARMC) Comm1LB.ci

HeapCentLB.decl.h HeapCentLB.def.h : HeapCentLB.ci charmxi
	$(CHARMC) HeapCentLB.ci

NeighborLB.decl.h NeighborLB.def.h : NeighborLB.ci charmxi
	$(CHARMC) NeighborLB.ci

WSLB.decl.h WS.def.h : WSLB.ci charmxi
	$(CHARMC) WSLB.ci

GreedyRefLB.decl.h GreedyRefLB.def.h : GreedyRefLB.ci charmxi
	$(CHARMC) GreedyRefLB.ci

RandRefLB.decl.h RandRefLB.def.h : RandRefLB.ci charmxi
	$(CHARMC) RandRefLB.ci

init.o: init.C $(CKHEADERS)
	$(CHARMC) -o init.o init.C

msgalloc.o: msgalloc.C $(CKHEADERS)
	$(CHARMC) -o msgalloc.o msgalloc.C

register.o: register.C $(CKHEADERS)
	$(CHARMC) -o register.o register.C

ckfutures.o: ckfutures.C $(CKHEADERS)
	$(CHARMC) -o ckfutures.o ckfutures.C

tempo.o: tempo.C $(CKHEADERS)
	$(CHARMC) -o tempo.o tempo.C

waitqd.o: waitqd.C $(CKHEADERS)
	$(CHARMC) -o waitqd.o waitqd.C

ckarray.o: ckarray.C $(CKHEADERS)
	$(CHARMC) -o ckarray.o ckarray.C

lbdb.o: lbdb.C $(CKHEADERS)
	$(CHARMC) -o lbdb.o lbdb.C

LBDBManager.o:	LBDBManager.C $(CKHEADERS)
	$(CHARMC) -o LBDBManager.o LBDBManager.C

LBComm.o:	LBComm.C $(CKHEADERS)
	$(CHARMC) -o LBComm.o LBComm.C

LBObj.o: LBObj.C $(CKHEADERS)
	$(CHARMC) -o LBObj.o LBObj.C

LBMachineUtil.o: LBObj.C $(CKHEADERS)
	$(CHARMC) -o LBMachineUtil.o LBMachineUtil.C

LBDatabase.o: LBDatabase.C $(CKHEADERS)
	$(CHARMC) -o LBDatabase.o LBDatabase.C

Refiner.o: Refiner.C $(CKHEADERS)
	$(CHARMC) -o Refiner.o Refiner.C

ObjGraph.o: ObjGraph.C $(CKHEADERS)
	$(CHARMC) -o ObjGraph.o ObjGraph.C

CentralLB.o: CentralLB.C $(CKHEADERS)
	$(CHARMC) -o CentralLB.o CentralLB.C

RandCentLB.o: RandCentLB.C $(CKHEADERS)
	$(CHARMC) -o RandCentLB.o RandCentLB.C

RecBisectBfLB.o: RecBisectBfLB.C $(CKHEADERS) graph.h fifoInt.h bitvecset.h
	$(CHARMC) -o RecBisectBfLB.o RecBisectBfLB.C

graph.o: graph.c 
	$(CHARMC) -o graph.o graph.c

fifoInt.o: fifoInt.c fifoInt.h
	$(CHARMC) -o fifoInt.o fifoInt.c

bitvecset.o: bitvecset.c bitvecset.h
	$(CHARMC) -o bitvecset.o bitvecset.c

MetisLB.o: MetisLB.C $(CKHEADERS)
	$(CHARMC) -o MetisLB.o MetisLB.C

RefineLB.o: RefineLB.C $(CKHEADERS)
	$(CHARMC) -o RefineLB.o RefineLB.C

CommLB.o: CommLB.C $(CKHEADERS)
	$(CHARMC) -o CommLB.o CommLB.C

Comm1LB.o: Comm1LB.C $(CKHEADERS)
	$(CHARMC) -o Comm1LB.o Comm1LB.C

HeapCentLB.o: HeapCentLB.C $(CKHEADERS)
	$(CHARMC) -o HeapCentLB.o HeapCentLB.C

Set.o: Set.C $(CKHEADERS) Set.h elements.h 
	$(CHARMC) -o Set.o Set.C

heap.o: heap.C $(CKHEADERS) elements.h heap.h
	$(CHARMC) -o heap.o heap.C

CommLBHeap.o: CommLBHeap.C $(CKHEADERS) CommLBHeap.h
	$(CHARMC) -o CommLBHeap.o CommLBHeap.C

NeighborLB.o: NeighborLB.C $(CKHEADERS)
	$(CHARMC) -o NeighborLB.o NeighborLB.C

WSLB.o: WSLB.C $(CKHEADERS)
	$(CHARMC) -o WSLB.o WSLB.C

GreedyRefLB.o: GreedyRefLB.C $(CKHEADERS)
	$(CHARMC) -o GreedyRefLB.o GreedyRefLB.C

RandRefLB.o: RandRefLB.C $(CKHEADERS)
	$(CHARMC) -o RandRefLB.o RandRefLB.C

qd.o: qd.C $(CKHEADERS)
	$(CHARMC) -o qd.o qd.C

main.o: main.C $(CKHEADERS)
	$(CHARMC) -o main.o main.C

debug-object.o: debug-object.h debug-object.C $(CKHEADERS)
	$(CHARMC) -o debug-object.o debug-object.C

debug-message.o: debug-message.C $(CKHEADERS)
	$(CHARMC) -o debug-message.o debug-message.C

info.o:	info.C $(CKHEADERS)
	$(CHARMC) -o info.o info.C

ck.o: ck.C $(CKHEADERS)
	$(CHARMC) -o ck.o ck.C

trace-none.o : trace-none.C trace.h $(CKHEADERS)
	$(CHARMC) -o trace-none.o trace-none.C

trace-projections.o:trace-projections.C trace-projections.h trace.h $(CKHEADERS)
	$(CHARMC) -o trace-projections.o trace-projections.C

trace-summary.o:trace-summary.C trace-summary.h trace.h $(CKHEADERS)
	$(CHARMC) -o trace-summary.o trace-summary.C


###############################################################################
#
# The CPM scanner
#
###############################################################################

conv-cpm: conv-cpm.o
	$(SEQCHARMC) -o conv-cpm -cp ../bin/ conv-cpm.o

conv-cpm.o: conv-cpm.c
	$(SEQCHARMC) conv-cpm.c

###############################################################################
#
# The interface translator
#
###############################################################################

XIHEADERS=xi-symbol.h xi-util.h xi-grammar.tab.h
CHARMXI=xi-main.o xi-symbol.o xi-grammar.tab.o xi-scan.o xi-util.o

charmxi: $(CHARMXI)
	$(SEQCHARMC) -language c++ -cp ../bin/ -o charmxi $(CHARMXI)

xi-main.o: xi-main.C $(XIHEADERS)
	$(SEQCHARMC) xi-main.C

xi-symbol.o: xi-symbol.C $(XIHEADERS)
	$(SEQCHARMC) xi-symbol.C

xi-util.o: xi-util.C $(XIHEADERS)
	$(SEQCHARMC) xi-util.C

xi-grammar.tab.o: xi-grammar.tab.C $(XIHEADERS)
	$(SEQCHARMC) xi-grammar.tab.C

xi-scan.o: xi-scan.C $(XIHEADERS)
	$(SEQCHARMC) xi-scan.C

###############################################################################
#
# Simple Messaging
#
###############################################################################

libtsm.a: converse simplemsg.c simplemsg.h
	$(CHARMC) -cp ../lib/ -o $@ simplemsg.c

libsm.a: converse sm.c sm.h
	$(CHARMC) -cp ../lib/ -o $@ sm.c

###############################################################################
#
# PVM
#
###############################################################################

PVMC=pvmc_buf.o pvmc_comm.o pvmc_conv.o pvmc_main.o pvmc_pack.o pvmc_groups.o

libpvmc.a: converse $(PVMC)
	$(CHARMC) -cp ../lib -o libpvmc.a $(PVMC)

pvmc_buf.o: pvmc_buf.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_buf.c

pvmc_conv.o: pvmc_conv.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_conv.c

pvmc_pack.o: pvmc_pack.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_pack.c

pvmc_comm.o: pvmc_comm.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_comm.c

pvmc_groups.o: pvmc_groups.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_groups.c

pvmc_main.o: pvmc_main.c pvmc.h pvm3.h
	$(CHARMC) -o $@ pvmc_main.c

libsdag.a: CDep.o
	$(CHARMC) -cp ../lib -o libsdag.a CDep.o

CDep.o: CDep.C CDep.h CMsgBuffer.h CWhenTrigger.h
	$(CHARMC) -c CDep.C

SDAGOBJ=sdag-main.o trans.o CLexer.o CParser.o CParsedFile.o CParseNode.o \
	sdag-globals.o CParseNodeProcess.o CEntry.o

sdagx: $(SDAGOBJ)
	$(SEQCHARMC) -cp ../bin -o sdagx -language c++ $(SDAGOBJ)

sdag-main.o: sdag-main.C CLexer.h CToken.h
	$(SEQCHARMC) -c sdag-main.C

trans.o: trans.c EToken.h
	$(SEQCHARMC) -c trans.c

CLexer.o: CLexer.C CLexer.h
	$(SEQCHARMC) -c CLexer.C

CParser.o: CParser.C CParser.h
	$(SEQCHARMC) -c CParser.C

CParsedFile.o: CParsedFile.C CParsedFile.h
	$(SEQCHARMC) -c CParsedFile.C

CParseNode.o: CParseNode.C CParseNode.h
	$(SEQCHARMC) -c CParseNode.C

sdag-globals.o: sdag-globals.C sdag-globals.h
	$(SEQCHARMC) -c sdag-globals.C

CParseNodeProcess.o: CParseNodeProcess.C CParseNode.h
	$(SEQCHARMC) -c CParseNodeProcess.C

CEntry.o: CEntry.C CEntry.h
	$(SEQCHARMC) -c CEntry.C

###############################################################################
#
# Charm++ IDL
#
###############################################################################

IDL_CFE_VERSION =       \"1.3.0\"
CPPFLAGS = -I. -DIDL_CFE_VERSION=$(IDL_CFE_VERSION)
CCFLAGS  = -g
IDLCHARMC = $(SEQCHARMC) $(CPPFLAGS) $(CCFLAGS)
RANLIB          = /bin/true
AR              = ar
ARFLAGS         = crv

drv_init.o: drv_init.cc
	$(IDLCHARMC) drv_init.cc

drv_private.o: drv_private.cc
	$(IDLCHARMC) drv_private.cc

drv_main.o: drv_main.cc
	$(IDLCHARMC) drv_main.cc

drv_args.o: drv_args.cc
	$(IDLCHARMC) drv_args.cc

drv_fork.o: drv_fork.cc
	$(IDLCHARMC) drv_fork.cc

drv_link.o: drv_link.cc
	$(IDLCHARMC) drv_link.cc

drv_preproc.o: drv_preproc.cc
	$(IDLCHARMC) drv_preproc.cc

libdrv.a: drv_init.o drv_private.o drv_main.o drv_args.o drv_fork.o \
          drv_link.o drv_preproc.o
	$(AR) $(ARFLAGS) libdrv.a drv_init.o drv_private.o drv_main.o \
		drv_args.o drv_fork.o drv_link.o drv_preproc.o
	$(RANLIB) $@

# Create the parser and lexer

fe_declarator.o: fe_declarator.cc
	$(IDLCHARMC) fe_declarator.cc

fe_private.o: fe_private.cc
	$(IDLCHARMC) fe_private.cc

fe_init.o: fe_init.cc
	$(IDLCHARMC) fe_init.cc

fe_extern.o: fe_extern.cc
	$(IDLCHARMC) fe_extern.cc

fe_interface_header.o: fe_interface_header.cc
	$(IDLCHARMC) fe_interface_header.cc

y.tab.o: y.tab.cc
	$(IDLCHARMC) y.tab.cc

lex.yy.o: lex.yy.cc
	$(IDLCHARMC) lex.yy.cc

libfe.a: fe_declarator.o fe_init.o fe_extern.o fe_interface_header.o  \
         y.tab.o lex.yy.o
	$(AR) $(ARFLAGS) libfe.a fe_declarator.o fe_init.o fe_extern.o \
		fe_interface_header.o y.tab.o lex.yy.o
	$(RANLIB) libfe.a

be_generator.o: be_generator.cc be.hh be_generator.hh be_classes.hh
	$(IDLCHARMC) be_generator.cc 

be_classes.o: be_classes.cc be.hh be_generator.hh be_classes.hh
	$(IDLCHARMC) be_classes.cc

be_init.o: be_init.cc be.hh be_generator.hh be_classes.hh
	$(IDLCHARMC) be_init.cc 

be_produce.o: be_produce.cc be.hh be_generator.hh be_classes.hh
	$(IDLCHARMC) be_produce.cc

be_args.o: be_args.cc be.hh be_generator.hh be_classes.hh
	$(IDLCHARMC) be_args.cc

libbe.a: be_generator.o be_classes.o be_init.o be_produce.o be_args.o
	$(AR) $(ARFLAGS) libbe.a be_generator.o be_classes.o \
		be_init.o be_produce.o be_args.o
	$(RANLIB) libbe.a

ast_array.o: ast_array.cc
	$(IDLCHARMC) ast_array.cc

ast_argument.o: ast_argument.cc
	$(IDLCHARMC) ast_argument.cc

ast_attribute.o: ast_attribute.cc
	$(IDLCHARMC) ast_attribute.cc

ast_check.o: ast_check.cc
	$(IDLCHARMC) ast_check.cc

ast_concrete_type.o: ast_concrete_type.cc
	$(IDLCHARMC) ast_concrete_type.cc

ast_constant.o: ast_constant.cc
	$(IDLCHARMC) ast_constant.cc

ast_decl.o: ast_decl.cc
	$(IDLCHARMC) ast_decl.cc

ast_enum.o: ast_enum.cc
	$(IDLCHARMC) ast_enum.cc

ast_enum_val.o: ast_enum_val.cc
	$(IDLCHARMC) ast_enum_val.cc

ast_exception.o: ast_exception.cc
	$(IDLCHARMC) ast_exception.cc

ast_expression.o: ast_expression.cc
	$(IDLCHARMC) ast_expression.cc

ast_field.o: ast_field.cc
	$(IDLCHARMC) ast_field.cc

ast_interface.o: ast_interface.cc
	$(IDLCHARMC)  ast_interface.cc

ast_interface_fwd.o: ast_interface_fwd.cc
	$(IDLCHARMC) ast_interface_fwd.cc

ast_module.o: ast_module.cc
	$(IDLCHARMC) ast_module.cc

ast_operation.o: ast_operation.cc
	$(IDLCHARMC) ast_operation.cc

ast_predefined_type.o: ast_predefined_type.cc
	$(IDLCHARMC) ast_predefined_type.cc

ast_root.o: ast_root.cc
	$(IDLCHARMC) ast_root.cc

ast_sequence.o: ast_sequence.cc
	$(IDLCHARMC) ast_sequence.cc

ast_string.o: ast_string.cc
	$(IDLCHARMC) ast_string.cc

ast_structure.o: ast_structure.cc
	$(IDLCHARMC) ast_structure.cc

ast_type.o: ast_type.cc
	$(IDLCHARMC) ast_type.cc

ast_typedef.o: ast_typedef.cc
	$(IDLCHARMC) ast_typedef.cc

ast_union.o: ast_union.cc
	$(IDLCHARMC) ast_union.cc

ast_union_branch.o: ast_union_branch.cc
	$(IDLCHARMC) ast_union_branch.cc

ast_union_label.o: ast_union_label.cc
	$(IDLCHARMC) ast_union_label.cc

ast_generator.o: ast_generator.cc
	$(IDLCHARMC) ast_generator.cc

ast_redef.o: ast_redef.cc
	$(IDLCHARMC) ast_redef.cc

ast_recursive.o: ast_recursive.cc
	$(IDLCHARMC) ast_recursive.cc

libast.a: ast_array.o ast_argument.o ast_attribute.o ast_check.o \
          ast_concrete_type.o ast_constant.o ast_decl.o ast_enum.o \
          ast_enum_val.o ast_exception.o ast_expression.o ast_field.o \
          ast_interface.o ast_interface_fwd.o ast_module.o ast_operation.o \
          ast_predefined_type.o ast_root.o ast_sequence.o ast_string.o \
          ast_structure.o ast_type.o ast_typedef.o ast_union.o \
          ast_union_branch.o ast_union_label.o ast_generator.o ast_redef.o \
          ast_recursive.o
	$(AR) $(ARFLAGS) libast.a ast_array.o ast_argument.o ast_attribute.o \
		ast_check.o ast_concrete_type.o ast_constant.o ast_decl.o \
		ast_enum.o ast_enum_val.o ast_exception.o ast_expression.o \
		ast_field.o ast_interface.o ast_interface_fwd.o ast_module.o \
		ast_operation.o ast_predefined_type.o ast_root.o \
		ast_sequence.o ast_string.o ast_structure.o ast_type.o \
		ast_typedef.o ast_union.o ast_union_branch.o \
		ast_union_label.o ast_generator.o ast_redef.o ast_recursive.o
	$(RANLIB) $@

utl_scope.o: utl_scope.cc
	$(IDLCHARMC) utl_scope.cc

utl_stack.o: utl_stack.cc
	$(IDLCHARMC) utl_stack.cc

utl_string.o: utl_string.cc
	$(IDLCHARMC) utl_string.cc

utl_decllist.o: utl_decllist.cc
	$(IDLCHARMC) utl_decllist.cc

utl_labellist.o: utl_labellist.cc
	$(IDLCHARMC) utl_labellist.cc

utl_namelist.o: utl_namelist.cc
	$(IDLCHARMC) utl_namelist.cc

utl_exceptlist.o: utl_exceptlist.cc
	$(IDLCHARMC) utl_exceptlist.cc

utl_strlist.o: utl_strlist.cc
	$(IDLCHARMC) utl_strlist.cc

utl_list.o: utl_list.cc
	$(IDLCHARMC) utl_list.cc

utl_exprlist.o: utl_exprlist.cc
	$(IDLCHARMC) utl_exprlist.cc

utl_error.o: utl_error.cc
	$(IDLCHARMC) utl_error.cc

utl_indenter.o: utl_indenter.cc
	$(IDLCHARMC) utl_indenter.cc

utl_identifier.o: utl_identifier.cc
	$(IDLCHARMC) utl_identifier.cc

utl_idlist.o: utl_idlist.cc
	$(IDLCHARMC) utl_idlist.cc

utl_global.o: utl_global.cc
	$(IDLCHARMC) utl_global.cc

libutil.a: utl_scope.o utl_stack.o utl_string.o utl_decllist.o \
           utl_labellist.o utl_namelist.o utl_exceptlist.o utl_strlist.o \
           utl_list.o utl_exprlist.o utl_error.o utl_indenter.o \
           utl_identifier.o utl_idlist.o utl_global.o
	$(AR) $(ARFLAGS) libutil.a  utl_scope.o utl_stack.o utl_string.o \
		utl_decllist.o utl_labellist.o utl_namelist.o \
		utl_exceptlist.o utl_strlist.o utl_list.o utl_exprlist.o \
		utl_error.o utl_indenter.o utl_identifier.o utl_idlist.o \
		utl_global.o
	$(RANLIB) libutil.a

nr_narrow.o: nr_narrow.cc
	$(IDLCHARMC) nr_narrow.cc

libnarrow.a: nr_narrow.o
	$(AR) $(ARFLAGS) libnarrow.a nr_narrow.o
	$(RANLIB) libnarrow.a

charmidl: libdrv.a libbe.a libfe.a libast.a libutil.a libnarrow.a
	$(IDLCHARMC) -o charmidl -language c++ -cp ../bin libdrv.a libbe.a \
		libfe.a libast.a libutil.a libnarrow.a

libidl.a: idl.o
	$(CHARMC) -cp ../lib/ -o libidl.a idl.o
	/bin/cp idl.decl.h ../include/

idl.o: basics idl.C idl.h idl.decl.h idl.def.h
	$(CHARMC) -c idl.C

idl.decl.h idl.def.h: charmxi idl.ci
	$(CHARMC) idl.ci
	/bin/cp idl.decl.h ../include/

###############################################################################
#
# Make clean
#
###############################################################################

clean:
	rm -f conv-host conv-host-notify conv-host-silent charmxi conv-cpm
	rm -f TAGS basics cmk_extras core
	rm -f core *.a
	rm -f core *.o
	rm -f core #*#
	rm -f core *~

veryclean:
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../src
	rm -rf ../pgms
	rm -rf ../include
	mv Makefile .Makefile
	rm -rf [a-z]* [A-Z]*
	mv .Makefile Makefile
	rm -rf .[a-z]* .[A-Z]*
