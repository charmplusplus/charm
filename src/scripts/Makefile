#############################################################################"
#
# This one Makefile serves all of converse and charm.  It includes
# rules for building:
#
#     * The CONVERSE libraries
#     * Charm++ runtime
#     * ITC++ Translator
#     * Languages
#     * Documentation
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

all: charm++ bluegene LIBS ampi f90charm tsm sm pvm sdag doc

charm++: converse libck.a charmxi

bluegene: charm++ libbluegene.a

for-namd: charm++ pvm AMPIC

tsm: converse libtsm.a

sm: converse libsm.a

pvm: converse libpvmc.a

sdag: charm++ sdagx libsdag.a

FEM: FEMC FEMF

FEMC: charm++ LIBS

FEMF: FEMC libconv-utilf.a

AMPI: AMPIC AMPIF

AMPIC: charm++ LIBS libampi.a

AMPIF: charm++ libconv-utilf.a LIBS libampif.a

f90charm: charm++ libf90charm.a

translators: sdagx charmxi conv-cpm

LIBS: CONVLIBS CHARMLIBS
	cd libs; make otherlibs OPTS='$(OPTS)'

test-all: all
	cd ../pgms ; make test-all  OPTS='$(OPTS)'

test-converse: converse
	cd ../pgms ; make test-converse  OPTS='$(OPTS)'

ps-doc: basics
	cd doc ; make ps

pdf-doc: basics
	cd doc ; make pdf

html-doc: basics
	cd doc ; make html

doc: basics
	cd doc ; make all

CHARMC=../bin/charmc $(CHARMOPTS) $(OPTS)

SEQCHARMC=../bin/charmc -seq $(CHARMOPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.sh)
#
###############################################################################

CVHEADERS=cpthreads.h converse.h conv-trace.h trace.h conv-ccs.h \
      conv-autoconfig.h conv-mach.h conv-mach.sh \
      CkFutures.decl.h CkArray.decl.h CkReduction.decl.h \
      LBDatabase.decl.h CentralLB.decl.h \
      RandCentLB.decl.h RecBisectBfLB.decl.h \
      MetisLB.decl.h RefineLB.decl.h \
      HeapCentLB.decl.h NborBaseLB.decl.h NeighborLB.decl.h \
      WSLB.decl.h GreedyRefLB.decl.h RandRefLB.decl.h \
      tempo.decl.h waitqd.decl.h \
	  CommLB.decl.h Comm1LB.decl.h  \
	  ampi.decl.h ampimain.decl.h BlueGene.decl.h

# The .c files are there to be #included by clients whole
# This is a bit unusual, but makes client linking simpler.
UTILHEADERS=pup.h pup_c.h ckhashtable.h cklists.h conv-lists.h \
	sockRoutines.h sockRoutines.c

CKHEADERS=ck.h ckstream.h envelope.h init.h qd.h charm.h charm++.h \
      ckfutures.h ckarray.h ckreduction.h $(UTILHEADERS) \
	  tempo.h waitqd.h LBDatabase.h lbdb.h \
	  LBDBManager.h LBComm.h LBOM.h LBObj.h LBMachineUtil.h \
	  Refiner.h ObjGraph.h heap.h elements.h CommLBHeap.h \
	  CentralLB.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h HeapCentLB.h CommLB.h Comm1LB.h \
	  MetisLB.h NborBaseLB.h \
	  NeighborLB.h WSLB.h GreedyRefLB.h RandRefLB.h \
      ampi.h ampif.h BlueGene.h \
	  $(CVHEADERS)

ALLHEADERS=$(CKHEADERS) \
    simplemsg.h sm.h pvmc.h pvm3.h sdag.h \
    CDep.h CCounter.h CMsgBuffer.h CWhenTrigger.h TList.h \
	graph.h bitvecset.h Set.h ampiimpl.h

basics: headers
	touch basics

# Headers is used only for the dependencies
headers: dirs+sources headerlinks $(ALLHEADERS)


headerlinks: dirs+sources Makefile
	rm -f ../bin/charmc ; cp charmc ../bin/
	chmod +x ../bin/charmc
	rm -rf ../include ; mkdir ../include
	./charmconfig
	-echo "Soft-linking headers..."
	-for hdr in $(ALLHEADERS) ; do test ! -f "../include/$$hdr" && ln -f -s "../tmp/$$hdr" ../include ; done
	touch headerlinks	

dirs+sources:
	rm -f .vdir .gdir
	(cd .. ; basename `pwd`) 			> .vdir
	(cd .. ; basename `pwd` | sed -e 's@-.*@@') 	> .gdir
	find . -type l -exec rm {} \;
	rm -rf QuickThreads
	rm -rf libs
	rm -rf ../bin ; mkdir ../bin
	rm -rf ../lib ; mkdir ../lib
	rm -rf ../pgms
	rm -rf ../doc ; mkdir ../doc
	../../src/Common/scripts/gatherflat ../../src/Common/scripts .
	./gatherflat ../../src/Common/conv-core        .
	./gatherflat ../../src/Common/conv-ldb         .
	./gatherflat ../../src/Common/ck-core          .
	./gatherflat ../../src/Common/util             .
	./gatherflat ../../src/Common/ck-perf          .
	./gatherflat ../../src/Common/ck-ldb           .
	./gatherflat ../../src/Common/langs/simplemsg  .
	./gatherflat ../../src/Common/langs/pvmc       .
	./gatherflat ../../src/Common/langs/ampi       .
	./gatherflat ../../src/Common/langs/bluegene   .
	./gatherflat ../../src/Common/langs/sdag/xlator       .
	./gatherflat ../../src/Common/langs/sdag/runtime      .
	./gatherflat ../../src/Common/langs/f90charm      .
	./gatherflat ../../src/Common/xlat-i           .
	./gatherflat ../../src/Common/xlatcpm          .
	./gathertree ../../src/QuickThreads QuickThreads
	./gathertree ../../src/Common/libs libs
	./gathertree ../../src/Common.`cat .gdir`      .
	./gatherflat ../../src/`cat .vdir`             .
	./gathertree ../../pgms ../pgms
	./gathertree ../../doc doc
	rm -f .vdir .gdir
	touch dirs+sources


###############################################################################
#
# Converse Libraries
#
###############################################################################

TRACELIBS=libtrace-none.a libtrace-projections.a libtrace-summary.a

CVLIBS=libconv-core.a libconv-cplus-n.a libconv-cplus-y.a libconv-util.a libldb-rand.o libldb-spray.o libldb-graph.o libldb-test.o $(TRACELIBS)

LIBCONV_CORE=threads.o convcore.o conv-conds.o queueing.o msgmgr.o memory.o cpm.o cpthreads.o futures.o cldb.o random.o debug-message.o debug-object.o info.o generate.o edgelist.o conv-ccs.o quiescence.o 

LIBCONV_UTIL=pup_util.o pup_xlater.o pup_c.o ckhashtable.o sockRoutines.o conv-lists.o

LIBCONV_UTILF=pup_f.o

converse: basics QuickThreads/libckqt.a $(CVLIBS) charmrun-target conv-cpm

charmrun-target: 
	if [ -d charmrun ] ; then ( cd charmrun ; make OPTS='$(OPTS)' ) ; fi
	if [ -f charmrun ] ; then ( cp charmrun ../bin ) ; fi

QuickThreads/libckqt.a:
	. ./conv-mach.sh;cd QuickThreads;./configure $$CMK_QT
	rm -f QuickThreads/libqt.a
	cd QuickThreads ; make qt  OPTS='$(OPTS)'
	cp QuickThreads/libqt.a ../lib/libckqt.a

CONVLIBS: converse
	cd libs; make convlibs OPTS='$(OPTS)'

libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCONV_CORE)

libconv-cplus-y.a: machine.c $(CVHEADERS)	
	$(CHARMC) -cp ../lib/ -o $@ -DFOR_CPLUS=1 machine.c

libconv-cplus-n.a: machine.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ -DFOR_CPLUS=0 machine.c

libconv-util.a: $(LIBCONV_UTIL)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCONV_UTIL)

libconv-utilf.a: $(LIBCONV_UTILF)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCONV_UTILF)

cpm.o: cpm.c $(CVHEADERS)
	$(CHARMC) cpm.c

threads.o: threads.c $(CVHEADERS)
	$(CHARMC) -IQuickThreads threads.c

convcore.o: convcore.c $(CVHEADERS)
	$(CHARMC) convcore.c

conv-ccs.o: conv-ccs.c $(CVHEADERS)
	$(CHARMC) conv-ccs.c

sockRoutines.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) sockRoutines.c

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

quiescence.o: quiescence.h $(CVHEADERS)
	$(CHARMC) quiescence.c

libldb-rand.o: cldb.rand.c ../../src/Common/conv-ldb/cldb.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ -cp ../lib/ -o $@ cldb.rand.c

libldb-graph.o: cldb.graph.c ../../src/Common/conv-ldb/cldb.graph.h $(CVHEADERS)
	$(CHARMC) -I../../src/Common/conv-ldb/ -cp ../lib/ -o $@ cldb.graph.c 

libldb-spray.o: cldb.spray.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cldb.spray.c

libldb-test.o: cldb.test.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cldb.test.c

pup_util.o: pup_util.C $(UTILHEADERS)
	$(CHARMC) -o pup_util.o pup_util.C

pup_xlater.o: pup_xlater.C $(UTILHEADERS)
	$(CHARMC) -o pup_xlater.o pup_xlater.C

pup_c.o: pup_c.C $(UTILHEADERS)
	$(CHARMC) -o pup_c.o pup_c.C

pup_f.o: pup_f.f90
	$(CHARMC) -c pup_f.f90
	$(CHARMC) -cpmod ../include pupmod.M

ckhashtable.o: ckhashtable.C ckhashtable.h
	$(CHARMC) -o ckhashtable.o ckhashtable.C

conv-lists.o: conv-lists.C $(UTILHEADERS)
	$(CHARMC) -o conv-lists.o conv-lists.C

###############################################################################
#
# Charm Libraries
#
###############################################################################

CK_LIBS_CORE=libck.a

LIBCK_CORE=init.o register.o qd.o ck.o main.o msgalloc.o ckfutures.o \
           ckarray.o ckreduction.o \
           tempo.o waitqd.o LBDatabase.o lbdb.o \
	   LBDBManager.o LBComm.o LBObj.o LBMachineUtil.o Refiner.o \
	   ObjGraph.o \
	   CentralLB.o RandCentLB.o \
	   RecBisectBfLB.o graph.o bitvecset.o \
	   MetisLB.o RefineLB.o Set.o CommLB.o Comm1LB.o\
           HeapCentLB.o heap.o NborBaseLB.o NeighborLB.o WSLB.o \
	   GreedyRefLB.o RandRefLB.o CommLBHeap.o

CHARMLIBS: charm++
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

ampi.decl.h ampi.def.h : ampi.ci charmxi
	$(CHARMC) ampi.ci

ampimain.decl.h ampimain.def.h : ampimain.ci charmxi
	$(CHARMC) ampimain.ci

CkFutures.decl.h CkFutures.def.h : ckfutures.ci charmxi
	$(CHARMC) ckfutures.ci

tempo.decl.h tempo.def.h : tempo.ci charmxi
	$(CHARMC) tempo.ci

waitqd.decl.h waitqd.def.h : waitqd.ci charmxi
	$(CHARMC) waitqd.ci

CentralLB.decl.h CentralLB.def.h : CentralLB.ci charmxi
	$(CHARMC) CentralLB.ci

LBDatabase.decl.h LBDatabase.def.h : LBDatabase.ci charmxi
	$(CHARMC) LBDatabase.ci

RandCentLB.decl.h RandCentLB.def.h : RandCentLB.ci charmxi
	$(CHARMC) RandCentLB.ci

RecBisectBfLB.decl.h RecBisectBfLB.def.h : RecBisectBfLB.ci charmxi
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

NborBaseLB.decl.h NborBaseLB.def.h : NborBaseLB.ci charmxi
	$(CHARMC) NborBaseLB.ci

NeighborLB.decl.h NeighborLB.def.h : NeighborLB.ci charmxi
	$(CHARMC) NeighborLB.ci

WSLB.decl.h WSLB.def.h : WSLB.ci charmxi
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

CkReduction.decl.h CkReduction.def.h: ckreduction.ci charmxi
	$(CHARMC) ckreduction.ci

ckreduction.o: ckreduction.C $(CKHEADERS)
	$(CHARMC) -o ckreduction.o ckreduction.C

CkArray.decl.h CkArray.def.h : ckarray.ci charmxi
	$(CHARMC) ckarray.ci

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

LBMachineUtil.o: LBMachineUtil.C $(CKHEADERS)
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

RecBisectBfLB.o: RecBisectBfLB.C $(CKHEADERS) graph.h bitvecset.h
	$(CHARMC) -o RecBisectBfLB.o RecBisectBfLB.C

graph.o: graph.c 
	$(CHARMC) -o graph.o graph.c

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

NborBaseLB.o: NborBaseLB.C $(CKHEADERS)
	$(CHARMC) -o NborBaseLB.o NborBaseLB.C

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
# BlueGene Emulator
#
###############################################################################

BlueGene.decl.h BlueGene.def.h : BlueGene.ci charmxi
	$(CHARMC) BlueGene.ci

libbluegene.a: BlueGene.C BlueGene.h BlueGene.decl.h BlueGene.def.h
	$(CHARMC) -c BlueGene.C
	$(CHARMC) -cp ../lib -o libbluegene.a BlueGene.o


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

libpvmc.a: $(PVMC)
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

libampi.a: ampi.C ampi.h ampiimpl.h ampi.decl.h ampi.def.h ampimain.decl.h \
        ampimain.def.h
	$(CHARMC) -DAMPI_FORTRAN=0 -c ampi.C
	$(CHARMC) -cp ../lib -o libampi.a ampi.o

libampif.a: ampi.C ampimain.C ampif.C ampi.h ampiimpl.h ampi.decl.h ampi.def.h \
            ampimain.decl.h ampimain.def.h
	$(CHARMC) -DAMPI_FORTRAN=1 -c ampi.C
	$(CHARMC) -DAMPI_FORTRAN=1 -c ampimain.C
	$(CHARMC) -DAMPI_FORTRAN=1 -c ampif.C
	$(CHARMC) -cp ../lib -o libampif.a ampimain.o ampi.o ampif.o

libf90charm.a: f90main.ci f90main.h f90main.C charmf.h charmf.C futil.f90
	$(CHARMC) f90main.ci
	$(CHARMC) -c -DAMPI_FORTRAN=1 f90main.C
	$(CHARMC) -c -DAMPI_FORTRAN=1 charmf.C
	$(CHARMC) -c futil.f90
	$(CHARMC) -cp ../lib -o libf90charm.a f90main.o charmf.o futil.o

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
# Make clean
#
###############################################################################

clean:
	rm -f charmxi conv-cpm
	rm -f TAGS basics cmk_extras core
	rm -f core *.a
	rm -f core *.o
	rm -f core #*#
	rm -f core *~
	if [ -d charmrun ] ; then ( cd charmrun; make clean ) ; fi
	( cd doc; make clean )
	( cd libs; make clean )

veryclean:
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../src
	rm -rf ../pgms
	rm -rf ../include
	rm -rf ../doc
	mv Makefile .Makefile
	rm -rf [a-z]* [A-Z]*
	mv .Makefile Makefile
	rm -rf .[a-z]* .[A-Z]*


