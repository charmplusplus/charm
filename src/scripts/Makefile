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
L=../lib
INC=../include

all: charm++ bluegene LIBS AMPI f90charm TSM SM PVM doc

charm++: converse
	$(MAKE) $(L)/libck.a loadbalancers charmxi default_libs

bgampi: bluegene AMPI

TSM: converse $(L)/libtsm.a

SM: converse $(L)/libsm.a

PVM: converse $(L)/libpvmc.a

AMR: charm++ LIBS
	cd libs/ck-libs/amr; make all OPTS='$(OPTS)'

FEM: charm++ CONVLIBS FEMC LIBS FEMF

FEMC: charm++ LIBS

FEMF: FEMC

MBLOCK: AMPI MBLOCKC MBLOCKF

MBLOCKC: charm++ LIBS

MBLOCKF: MBLOCKC

AMPI: 
	$(MAKE) charm++ 
	$(MAKE) CONVLIBS 
	$(MAKE) AMPIC AMPIF

AMPIC: charm++ LIBS

AMPIF: AMPIC

f90charm: charm++ $(L)/libf90charm.a

default_libs: commlib multicast

multicast: $(L)/libck.a
	cd libs/ck-libs/multicast && $(MAKE)

commlib: $(L)/libck.a
	cd libs/ck-libs/commlib && $(MAKE)

translators: charmxi conv-cpm

pose:
	cd libs/ck-libs/pose && $(MAKE) 

LIBS: CONVLIBS CHARMLIBS
	cd libs; $(MAKE) otherlibs 

test-all: test

test: test-charm test-AMPI test-fem
	@echo "--- All tests passed ---"

test-converse: converse
	cd ../pgms ; make test-converse  OPTS='$(OPTS)'

test-charm: charm++
	cd ../pgms ; make test-charm  OPTS='$(OPTS)'

test-AMPI: AMPI
	cd ../pgms ; make test-AMPI  OPTS='$(OPTS)'

test-fem: FEM
	cd ../pgms ; make test-fem  OPTS='$(OPTS)'

ps-doc: basics
	cd ../doc ; make ps

pdf-doc: basics
	cd ../doc ; make pdf

html-doc: basics
	cd ../doc ; make html

doc: basics
	cd ../doc ; make doc

CHARMC=../bin/charmc $(CHARMOPTS) $(OPTS)

SEQCHARMC=../bin/charmc -seq $(CHARMOPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.sh)
#
###############################################################################

CVHEADERS=cpthreads.h converse.h conv-trace.h conv-random.h conv-qd.h \
      queueing.h conv-cpath.h conv-cpm.h persistent.h\
      trace.h trace-bluegene.h trace-projections.h charm-api.h \
      conv-ccs.h ccs-client.c ccs-client.h CcsServer.java\
      ccs-server.h ccs-auth.c ccs-auth.h \
      memory-isomalloc.h debug-conv.h debug-charm.h conv-autoconfig.h \
      conv-common.h conv-config.sh conv-config.h conv-mach.h conv-mach.sh \
      CkMarshall.decl.h CkArray.decl.h CkLocation.decl.h CkReduction.decl.h CkArrayReductionMgr.decl.h\
      CkFutures.decl.h LBDatabase.decl.h CentralLB.decl.h NullLB.decl.h CkCheckpoint.decl.h\
      RandCentLB.decl.h RecBisectBfLB.decl.h BaseLB.decl.h \
      MetisLB.decl.h DummyLB.decl.h RefineLB.decl.h OrbLB.decl.h \
      GreedyLB.decl.h NborBaseLB.decl.h NeighborLB.decl.h \
      WSLB.decl.h GreedyRefLB.decl.h RandRefLB.decl.h \
      tempo.decl.h waitqd.decl.h \
      CommLB.decl.h Comm1LB.decl.h  \
      charisma.decl.h \
      TraceSummary.decl.h	\
      BlueGene.decl.h blue.h

# The .c files are there to be #included by clients whole
# This is a bit unusual, but makes client linking simpler.
UTILHEADERS=pup.h pupf.h pup_c.h pup_stl.h pup_toNetwork4.h pup_paged.h \
	ckimage.h ckdll.h ckhashtable.h ckbitvector.h cklists.h \
	ckstatistics.h ckvector3d.h conv-lists.h \
	sockRoutines.h sockRoutines.c

CKHEADERS=ck.h ckstream.h envelope.h init.h qd.h charm.h charm++.h \
	  ckfutures.h \
	  ckcallback.h CkCallback.decl.h ckcallback-ccs.h 	\
	  ckarrayreductionmgr.h cknodegroupreduction.h cksection.h \
	  ckarray.h cklocation.h ckreduction.h ckcheckpoint.h\
	  readonly.h charisma.h \
	  $(UTILHEADERS) \
	  tempo.h waitqd.h LBDatabase.h lbdb.h \
	  LBDBManager.h LBComm.h LBOM.h LBObj.h LBMachineUtil.h \
	  Refiner.h ObjGraph.h heap.h elements.h CommLBHeap.h \
	  BaseLB.h CentralLB.h CentralLBMsg.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h OrbLB.h GreedyLB.h CommLB.h Comm1LB.h manager.h \
	  MetisLB.h NborBaseLB.h \
	  NeighborLB.h WSLB.h GreedyRefLB.h RandRefLB.h \
          ComlibStrategy.h ComlibArrayListener.h \
	  BlueGene.h middle.h middle-conv.h middle-blue.h \
	  $(CVHEADERS)

ALLHEADERS=$(CKHEADERS) \
    simplemsg.h sm.h pvmc.h pvm3.h sdag.h \
	graph.h bitvecset.h Set.h LBSimulation.h  \
	$(BLUE_HEADERS)

basics: headers
	touch basics

# Headers is used only for the dependencies
headers: dirs+sources 
	$(MAKE) headerlinks

conv-autoconfig.h: configure
	./configure

headerlinks: dirs+sources Makefile $(ALLHEADERS)
	@echo "Soft-linking headers..."
	-@for hdr in $(ALLHEADERS) ; do test ! -f "../include/$$hdr" && ./system_ln "../tmp/$$hdr" ../include ; done
	touch headerlinks

dirs+sources:
	rm -f .gdir
	(cd .. ; basename "`pwd`" | sed -e 's@-.*@@') 	> .gdir
	pwd
	find . -type l -exec rm {} \;
	rm -rf QuickThreads
	rm -rf libs
	rm -rf ../bin ; mkdir ../bin
	rm -rf ../lib ; mkdir ../lib
	rm -rf ../lib_so ; mkdir ../lib_so
	rm -rf ../pgms
	rm -rf ../doc ; ln -s ../doc ../doc
	../../src/scripts/gatherflat ../../src/scripts .
	./gatherflat ../../src/conv-core        .
	./gatherflat ../../src/conv-ldb         .
	./gatherflat ../../src/conv-ccs         .
	./gatherflat ../../src/conv-perf	.
	./gatherflat ../../src/ck-core          .
	./gatherflat ../../src/util             .
	./gatherflat ../../src/ck-perf          .
	./gatherflat ../../src/ck-ldb           .
	./gatherflat ../../src/langs/simplemsg  .
	./gatherflat ../../src/langs/pvmc       .
	./gatherflat ../../src/langs/bluegene   .
	./gatherflat ../../src/langs/f90charm   .
	./gatherflat ../../src/xlat-i           .
	./gatherflat ../../src/xlatcpm          .
	./gathertree ../../src/QuickThreads QuickThreads
	./gathertree ../../src/libs libs
	./gathertree ../../src/arch/util      .
	./gathertree ../../src/langs langs
	./gathertree ../../src/arch/common           .
	./gathertree ../../src/arch/`cat .gdir`      .
	./gatherflat ../../src/arch/`cat .vdir`      .
	./gathertree ../../pgms ../pgms
	rm -f ../bin/charmc ; cp charmc ../bin/
	rm -f ../bin/dep.pl ; cp dep.pl ../bin/
	chmod +x ../bin/charmc
	rm -f .gdir
	rm -rf ../include ; mkdir ../include
	-./system_ln  ../tmp/conv-*.*h ../include
	-./system_ln  ../tmp/cc-*.*h ../include
	if [ ! -f conv-common.h ] ; then ( touch conv-common.h ) ; fi
	touch dirs+sources

###############################################################################
#
# Converse Libraries
#
###############################################################################
CLBLIBS=$(L)/libldb-rand.o $(L)/libldb-spray.o $(L)/libldb-graph.o $(L)/libldb-test.o $(L)/libldb-bluegene.o
TRACELIBS=$(L)/libtrace-projections.a  $(L)/libtrace-summary.a \
          $(L)/libtrace-counter.a $(L)/libtrace-bluegene.a \
	  $(L)/libtrace-projector.a $(L)/libtrace-all.a $(L)/libtrace-converse.a
MEMLIBS=$(L)/libmemory-default.o $(L)/libmemory-os.o $(L)/libmemory-gnu.o \
          $(L)/libmemory-gnuold.o $(L)/libmemory-verbose.o 	\
	  $(L)/libmemory-paranoid.o \
	  $(L)/libmemory-leak.o $(L)/libmemory-isomalloc.o
THREADLIBS=$(L)/libthreads-default.o $(L)/libthreads-qt.o 	\
 	  $(L)/libthreads-context.o \
	  $(L)/libthreads-pthreads.o $(L)/libthreads-fibers.o

CVLIBS=$(L)/libconv-core.a \
	$(L)/libconv-cplus-y.a $(L)/libconv-cplus-n.a \
	$(L)/libconv-util.a $(L)/libconv-utilf.a \
	$(CLBLIBS) $(TRACELIBS) $(MEMLIBS) $(THREADLIBS) \
	$(L)/libccs-client.a

LIBCONV_CORE= convcore.o conv-conds.o queueing.o msgmgr.o \
	cpm.o cpthreads.o futures.o cldb.o random.o traceCore.o traceCoreCommon.o \
	converseProjections.o machineProjections.o \
	debug-conv.o generate.o edgelist.o conv-ccs.o ccs-builtins.o \
	quiescence.o isomalloc.o \
	global-nop.o

LIBCONV_UTIL=pup_util.o pup_xlater.o pup_c.o pup_paged.o\
	ckimage.o ckdll.o ckhashtable.o ckbitvector.o sockRoutines.o \
	conv-lists.o

LIBCONV_UTILF=pup_f.o

converse: basics 
	$(MAKE) converse-target

converse-target: QuickThreads/libqt.a $(CVLIBS) fmain-ok charmrun-target swapglobal-target conv-cpm

charmrun-target: daemon-target sockRoutines.c sockRoutines.h ccs-server.c ccs-server.h
	if [ -d charmrun ] ; then ( cd charmrun ; $(MAKE) OPTS='$(OPTS)' ) ; fi
	if [ -f charmrun ] ; then ( cp charmrun ../bin ) ; fi

daemon-target: sockRoutines-seq.o
	if test -f daemon.c ; then $(CHARMC) -seq daemon.c \
		sockRoutines-seq.o -o ../bin/charmd; fi
	if test -f daemon.c ; then $(CHARMC) -DFACELESS -seq daemon.c \
		sockRoutines-seq.o -o ../bin/charmd_faceless; fi
	touch daemon-target

fmain-ok: fmain.f90
	-@$(CHARMC) -c fmain.f90
	-@cp fmain.o ../lib
	-@touch fmain-ok

QuickThreads/libqt.a:  $(CVHEADERS)
	cd QuickThreads;./configure
	rm -f QuickThreads/libqt.a
	cd QuickThreads ; $(MAKE) qt  OPTS='$(OPTS)'
	cp QuickThreads/libqt.a ../lib/libckqt.a
	-(CHARMINC=.; . ./conv-config.sh; \
	  cp QuickThreads/libqt.$$CMK_SHARED_SUF ../lib_so && \
	  cp QuickThreads/libqt.$$CMK_SHARED_SUF ../lib_so/libckqt.$$CMK_SHARED_SUF)

METIS:  charm++
	cd libs/conv-libs/metis; $(MAKE) OPTS='$(OPTS)'

CONVLIBS: converse
	cd libs; $(MAKE) convlibs OPTS='$(OPTS)'

$(L)/libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) -o $@ $(LIBCONV_CORE)

$(L)/libconv-cplus-y.a: $(L)/libconv-cplus-n.a

$(L)/libconv-cplus-n.a: machine.c $(CVHEADERS)
	@-test -f $(INC)/mpi.h && mv -f $(INC)/mpi.h $(INC)/mpi.BAK || true
	$(CHARMC) -o $(L)/libconv-cplus-y.a -DFOR_CPLUS=1 machine.c
	$(CHARMC) -o $(L)/libconv-cplus-n.a -DFOR_CPLUS=0 machine.c
	@-test -f $(INC)/mpi.BAK && mv -f $(INC)/mpi.BAK $(INC)/mpi.h || true

$(L)/libconv-util.a: $(LIBCONV_UTIL)
	$(CHARMC) -o $@ $(LIBCONV_UTIL)

$(L)/libconv-utilf.a: $(LIBCONV_UTILF)
	-$(CHARMC) -o $@ $(LIBCONV_UTILF)

$(L)/libccs-client.a: ccs-client.o
	$(CHARMC) -o $@ ccs-client.o

cpm.o: cpm.c $(CVHEADERS)
	$(CHARMC) cpm.c

isomalloc.o: isomalloc.c $(CVHEADERS)
	$(CHARMC) isomalloc.c

convcore.o: convcore.c $(CVHEADERS)
	$(CHARMC) convcore.c

conv-ccs.o: conv-ccs.c $(CVHEADERS)
	$(CHARMC) conv-ccs.c

ccs-builtins.o: ccs-builtins.C $(CVHEADERS)
	$(CHARMC) ccs-builtins.C

ccs-client.o: ccs-client.c $(CVHEADERS)
	$(CHARMC) ccs-client.c

sockRoutines.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -DCMK_USE_CONVERSE=1 sockRoutines.c

sockRoutines-seq.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -seq -o sockRoutines-seq.o sockRoutines.c

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

debug-conv.o: debug-conv.c $(CVHEADERS)
	$(CHARMC) debug-conv.c

edgelist.o : edgelist.c typedefs.h $(CVHEADERS)
	$(CHARMC) edgelist.c

generate.o : generate.c typedefs.h $(CVHEADERS)
	$(CHARMC) generate.c

cldb.o: cldb.c cldb.h $(CVHEADERS)
	$(CHARMC) cldb.c

quiescence.o: quiescence.c $(CVHEADERS)
	$(CHARMC) quiescence.c

pup_util.o: pup_util.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_util.o pup_util.C

pup_xlater.o: pup_xlater.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_xlater.o pup_xlater.C

pup_c.o: pup_c.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_c.o pup_c.C

pup_paged.o: pup_paged.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_paged.o pup_paged.C
	
pup_f.o: pup_f.f90 $(CVHEADERS)
	-$(CHARMC) -c pup_f.f90
	-$(CHARMC) -cpmod ../include pupmod.M

ckhashtable.o: ckhashtable.C ckhashtable.h $(CVHEADERS)
	$(CHARMC) -o ckhashtable.o ckhashtable.C

ckbitvector.o: ckbitvector.C ckbitvector.h $(CVHEADERS)
	$(CHARMC) -o ckbitvector.o ckbitvector.C

ckdll.o: ckdll.C ckdll.h $(CVHEADERS)
	$(CHARMC) -o ckdll.o ckdll.C

ckimage.o: ckimage.C ckimage.h $(CVHEADERS)
	$(CHARMC) -o ckimage.o ckimage.C

conv-lists.o: conv-lists.C $(UTILHEADERS) $(CVHEADERS)
	$(CHARMC) -o conv-lists.o conv-lists.C

## Converse load balancers (seed balancers, -balance)
$(L)/libldb-rand.o: cldb.rand.c cldb.h $(CVHEADERS)
	$(CHARMC) -o $@ cldb.rand.c

$(L)/libldb-graph.o: cldb.graph.c graph.h $(CVHEADERS)
	$(CHARMC) -o $@ cldb.graph.c

$(L)/libldb-spray.o: cldb.spray.c $(CVHEADERS)
	$(CHARMC) -o $@ cldb.spray.c

$(L)/libldb-test.o: cldb.test.c $(CVHEADERS)
	$(CHARMC) -o $@ cldb.test.c

$(L)/libldb-bluegene.o: cldb.bluegene.c $(CVHEADERS)
	$(CHARMC) -o $@ cldb.bluegene.c

## Memory allocation libraries (-memory)
MEM_DEPS=memory.c memory-gnu.c memory-gnuold.c converse.h conv-mach.h $(CVHEADERS)
$(L)/libmemory-default.o: $(MEM_DEPS)
	$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_DEFAULT memory.c

$(L)/libmemory-os.o: $(MEM_DEPS)
	$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_OS memory.c

# If the system doesn't have sbrk, these compilations may fail.
#  This is OK, but then we can't use "-memory gnu" or friends.
$(L)/libmemory-gnu.o: $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_GNU memory.c

$(L)/libmemory-gnuold.o: $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_GNUOLD memory.c

$(L)/libmemory-verbose.o: memory-verbose.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_VERBOSE memory.c

$(L)/libmemory-paranoid.o: memory-paranoid.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_PARANOID memory.c

$(L)/libmemory-leak.o: memory-leak.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_LEAK memory.c

$(L)/libmemory-cache.o: memory-cache.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_CACHE memory.c

$(L)/libmemory-isomalloc.o: memory-isomalloc.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_ISOMALLOC memory.c

## Thread libraries (-thread)
$(L)/libthreads-default.o: threads.c $(CVHEADERS)
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_DEFAULT=1 -IQuickThreads threads.c

$(L)/libthreads-qt.o: threads.c $(CVHEADERS)
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_QT=1 -IQuickThreads threads.c

$(L)/libthreads-context.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_CONTEXT=1 -IQuickThreads threads.c

$(L)/libthreads-pthreads.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_PTHREADS=1 -IQuickThreads threads.c

$(L)/libthreads-fibers.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_FIBERS=1 -IQuickThreads threads.c 2> /dev/null

## Global swapping (-swapglobal)
global-nop.o: global-nop.c $(CVHEADERS)
	$(CHARMC) -c global-nop.c -o $@

swapglobal-target: $(L)/libglobal-swap.o

$(L)/libglobal-swap.o: global-elfgot.C $(CVHEADERS)
	-$(CHARMC) -c global-elfgot.C -o $@


###############################################################################
#
# Charm Libraries
#
###############################################################################

CK_LIBS_CORE=$(L)/libck.a

LIBCK_CORE=trace-common.o init.o register.o qd.o ck.o main.o msgalloc.o \
	   ckfutures.o debug-message.o ckcallback.o \
	   cklocation.o ckarray.o ckreduction.o ckarrayreductionmgr.o\
           tempo.o waitqd.o LBDatabase.o lbdb.o charisma.o ckcheckpoint.o\
           LBDBManager.o LBComm.o LBObj.o LBMachineUtil.o \
	   BaseLB.o CentralLB.o NborBaseLB.o WSLB.o ObjGraph.o graph.o  \
           Refiner.o bitvecset.o Set.o heap.o CommLBHeap.o \
	   NullLB.o LBSimulation.o 	\
	   ComlibStrategy.o ComlibArrayListener.o  \
	   charmProjections.o

CHARMLIBS: charm++
	cd libs; $(MAKE) charmlibs OPTS='$(OPTS)'

$(L)/libck.a: $(LIBCK_CORE)
	$(CHARMC) -o $@ $(LIBCK_CORE)

CkFutures.decl.h CkFutures.def.h : ckfutures.ci charmxi
	$(CHARMC) ckfutures.ci

tempo.decl.h tempo.def.h : tempo.ci charmxi
	$(CHARMC) tempo.ci

waitqd.decl.h waitqd.def.h : waitqd.ci charmxi
	$(CHARMC) waitqd.ci

LBDatabase.decl.h LBDatabase.def.h : LBDatabase.ci charmxi
	$(CHARMC) LBDatabase.ci

NullLB.decl.h NullLB.def.h : NullLB.ci charmxi
	$(CHARMC) NullLB.ci

BaseLB.decl.h BaseLB.def.h : BaseLB.ci charmxi
	$(CHARMC) BaseLB.ci

TraceSummary.decl.h TraceSummary.def.h: trace-summary.ci charmxi
	$(CHARMC) trace-summary.ci

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

qd.o: qd.C $(CKHEADERS)
	$(CHARMC) -o qd.o qd.C

main.o: main.C $(CKHEADERS)
	$(CHARMC) -o main.o main.C

debug-message.o: debug-message.C $(CKHEADERS)
	$(CHARMC) -I. debug-message.C

ck.o: ck.C $(CKHEADERS)
	$(CHARMC) -o ck.o ck.C

waitqd.o: waitqd.C $(CKHEADERS)
	$(CHARMC) -o waitqd.o waitqd.C

CkReduction.decl.h CkReduction.def.h: ckreduction.ci charmxi
	$(CHARMC) ckreduction.ci

ckreduction.o: ckreduction.C $(CKHEADERS)
	$(CHARMC) -o ckreduction.o ckreduction.C

CkCheckpoint.decl.h CkCheckpoint.def.h: ckcheckpoint.ci charmxi
	$(CHARMC) ckcheckpoint.ci

ckcheckpoint.o: ckcheckpoint.C $(CKHEADERS)
	$(CHARMC) -o ckcheckpoint.o ckcheckpoint.C

CkMarshall.decl.h CkMarshall.def.h : ckmarshall.ci charmxi
	$(CHARMC) ckmarshall.ci

CkArrayReductionMgr.decl.h CkArrayReductionMgr.def.h : ckarrayreductionmgr.ci charmxi
	$(CHARMC) ckarrayreductionmgr.ci

ckarrayreductionmgr.o : ckarrayreductionmgr.C $(CKHEADERS)
	$(CHARMC) -o ckarrayreductionmgr.o ckarrayreductionmgr.C


CkArray.decl.h CkArray.def.h : ckarray.ci charmxi
	$(CHARMC) ckarray.ci

ckarray.o: ckarray.C $(CKHEADERS)
	$(CHARMC) -o ckarray.o ckarray.C


CkLocation.decl.h: cklocation.ci charmxi
	$(CHARMC) cklocation.ci

cklocation.o: cklocation.C $(CKHEADERS)
	$(CHARMC) -c cklocation.C

ckcallback.o: ckcallback.C $(CKHEADERS) CkCallback.decl.h
	$(CHARMC) -c ckcallback.C

CkCallback.decl.h: ckcallback.ci charmxi
	$(CHARMC) ckcallback.ci

charisma.decl.h charisma.def.h : charisma.ci charmxi
	$(CHARMC) charisma.ci

charisma.o: charisma.C $(CKHEADERS)
	$(CHARMC) -o charisma.o charisma.C

## Charm++ Load balancer (Dynamic load balancer)

BaseLB.o: BaseLB.C $(CKHEADERS)
	$(CHARMC) -c BaseLB.C

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

NullLB.o: NullLB.C $(CKHEADERS)
	$(CHARMC) -c NullLB.C

Refiner.o: Refiner.C $(CKHEADERS)
	$(CHARMC) -o Refiner.o Refiner.C

ObjGraph.o: ObjGraph.C $(CKHEADERS)
	$(CHARMC) -o ObjGraph.o ObjGraph.C

graph.o: graph.c  $(CVHEADERS)
	$(CHARMC) -o graph.o graph.c

bitvecset.o: bitvecset.c bitvecset.h $(CVHEADERS)
	$(CHARMC) -o bitvecset.o bitvecset.c

Set.o: Set.C $(CKHEADERS) Set.h elements.h
	$(CHARMC) -o Set.o Set.C

heap.o: heap.C $(CKHEADERS) elements.h heap.h
	$(CHARMC) -o heap.o heap.C

CentralLB.decl.h: CentralLB.ci charmxi
	$(CHARMC) CentralLB.ci

CentralLB.o: CentralLB.C CentralLB.decl.h $(CKHEADERS)
	$(CHARMC) -c CentralLB.C

NborBaseLB.decl.h: NborBaseLB.ci charmxi
	$(CHARMC) NborBaseLB.ci

NborBaseLB.o: NborBaseLB.C NborBaseLB.decl.h $(CKHEADERS)
	$(CHARMC) -c NborBaseLB.C

CommLBHeap.o: CommLBHeap.C $(CKHEADERS)
	$(CHARMC) -c CommLBHeap.C

LBSimulation.o : LBSimulation.h LBSimulation.C $(CVHEADERS)
	$(CHARMC) -c LBSimulation.C

## Communication Library array listener code
ComlibArrayListener.o : ComlibArrayListener.C ComlibArrayListener.h $(CVHEADERS)
	$(CHARMC) -c ComlibArrayListener.C

ComlibStrategy.o : ComlibStrategy.C ComlibStrategy.h $(CVHEADERS)
	$(CHARMC) -c ComlibStrategy.C


## Tracing libraries (profiling, -tracemode)
LIBTRACE_PROJ=trace-projections.o trace-common.o 
$(L)/libtrace-projections.a: $(LIBTRACE_PROJ)
	$(CHARMC) -o $@ $(LIBTRACE_PROJ)

LIBTRACE_SUMM=trace-summary.o trace-common.o
$(L)/libtrace-summary.a: $(LIBTRACE_SUMM)
	$(CHARMC) -o $@ $(LIBTRACE_SUMM)

LIBTRACE_COUNT=trace-counter.o trace-common.o
$(L)/libtrace-counter.a: $(LIBTRACE_COUNT)
	$(CHARMC) -o $@ $(LIBTRACE_COUNT)

LIBTRACE_BG=trace-bluegene.o trace-common.o
$(L)/libtrace-bluegene.a:$(LIBTRACE_BG)
	$(CHARMC) -o $@ $(LIBTRACE_BG)

LIBTRACE_ALL=trace-all.o trace-projections.o trace-summary.o trace-common.o trace-projector.o traceCore.o traceCoreCommon.o charmProjections.o converseProjections.o machineProjections.o 
$(L)/libtrace-all.a: $(LIBTRACE_ALL)
	$(CHARMC) -o $@ $(LIBTRACE_ALL)
	
LIBTRACE_PROJECTOR=trace-projector.o trace-common.o traceCore.o traceCoreCommon.o charmProjections.o converseProjections.o machineProjections.o 
$(L)/libtrace-projector.a: $(LIBTRACE_PROJECTOR)
	$(CHARMC) -o  $@ $(LIBTRACE_PROJECTOR)

LIBTRACE_CONVERSE=trace-converse.o
$(L)/libtrace-converse.a: $(LIBTRACE_CONVERSE)
	$(CHARMC) -o  $@ $(LIBTRACE_CONVERSE)

trace-common.o: trace-common.C $(CKHEADERS)
	$(CHARMC) trace-common.C

trace-converse.o: trace-converse.c $(CKHEADERS)
	$(CHARMC) trace-converse.c

trace-projections.o: trace-projections.C trace-projections.h trace-common.h  $(CKHEADERS)
	$(CHARMC) -o trace-projections.o trace-projections.C

trace-projector.o: trace-projector.C trace-projector.h $(CKHEADERS) 
	$(CHARMC) -o trace-projector.o trace-projector.C
	
trace-summary.o:trace-summary.C trace-summary.h trace-common.h trace-summaryBOC.h $(CKHEADERS)
	$(CHARMC) -o trace-summary.o trace-summary.C

trace-counter.o:trace-counter.C trace-counter.h trace-common.h $(CKHEADERS)
	$(CHARMC) -o trace-counter.o trace-counter.C

trace-bluegene.o: trace-bluegene.C trace-bluegene.h $(CKHEADERS) $(BLUE_HEADERS)
	$(CHARMC) -o trace-bluegene.o trace-bluegene.C

traceCore.o: traceCore.C $(CVHEADERS)
	$(CHARMC) traceCore.C

traceCoreCommon.o: traceCoreCommon.C $(CVHEADERS)
	$(CHARMC) traceCoreCommon.C

charmProjections.o: charmProjections.C $(CVHEADERS)
	$(CHARMC) charmProjections.C

converseProjections.o: converseProjections.C $(CVHEADERS)
	$(CHARMC) converseProjections.C

machineProjections.o: machineProjections.C $(CVHEADERS)
	$(CHARMC) machineProjections.C

trace-all.o:trace-all.C $(CVHEADERS)
	$(CHARMC) -o trace-all.o trace-all.C

###############################################################################
#
# Dynamic Load Balancers
#
###############################################################################

# Automatically generated by script Makefile_lb.sh
#  on Sun Feb 16 13:16:07 EST 2003
LOADBALANCERS=\
   $(L)/libmoduleDummyLB.a \
   $(L)/libmoduleRandCentLB.a \
   $(L)/libmoduleRecBisectBfLB.a \
   $(L)/libmoduleMetisLB.a \
   $(L)/libmoduleRefineLB.a \
   $(L)/libmoduleCommLB.a \
   $(L)/libmoduleComm1LB.a \
   $(L)/libmoduleGreedyLB.a \
   $(L)/libmoduleNeighborLB.a \
   $(L)/libmoduleGreedyRefLB.a \
   $(L)/libmoduleOrbLB.a \
   $(L)/libmoduleRandRefLB.a \
   $(L)/libmoduleWSLB.a \
   manager.o

manager.o: manager.C manager.h $(CVHEADERS)
	$(CHARMC) -c manager.C

DummyLB.decl.h: DummyLB.ci charmxi
	$(CHARMC) DummyLB.ci

DummyLB.o: DummyLB.C DummyLB.decl.h $(CKHEADERS)
	$(CHARMC) -c DummyLB.C

$(L)/libmoduleDummyLB.a: DummyLB.o 
	$(CHARMC) -o $(L)/libmoduleDummyLB.a DummyLB.o 
	

RandCentLB.decl.h: RandCentLB.ci charmxi
	$(CHARMC) RandCentLB.ci

RandCentLB.o: RandCentLB.C RandCentLB.decl.h $(CKHEADERS)
	$(CHARMC) -c RandCentLB.C

$(L)/libmoduleRandCentLB.a: RandCentLB.o 
	$(CHARMC) -o $(L)/libmoduleRandCentLB.a RandCentLB.o 
	

RecBisectBfLB.decl.h: RecBisectBfLB.ci charmxi
	$(CHARMC) RecBisectBfLB.ci

RecBisectBfLB.o: RecBisectBfLB.C RecBisectBfLB.decl.h $(CKHEADERS)
	$(CHARMC) -c RecBisectBfLB.C

$(L)/libmoduleRecBisectBfLB.a: RecBisectBfLB.o 
	$(CHARMC) -o $(L)/libmoduleRecBisectBfLB.a RecBisectBfLB.o 
	

MetisLB.decl.h: MetisLB.ci charmxi
	$(CHARMC) MetisLB.ci

MetisLB.o: MetisLB.C MetisLB.decl.h $(CKHEADERS)
	$(CHARMC) -c MetisLB.C

$(L)/libmoduleMetisLB.a: MetisLB.o 
	$(CHARMC) -o $(L)/libmoduleMetisLB.a MetisLB.o 
	cp libmoduleMetisLB.dep $(L)/

RefineLB.decl.h: RefineLB.ci charmxi
	$(CHARMC) RefineLB.ci

RefineLB.o: RefineLB.C RefineLB.decl.h $(CKHEADERS)
	$(CHARMC) -c RefineLB.C

$(L)/libmoduleRefineLB.a: RefineLB.o 
	$(CHARMC) -o $(L)/libmoduleRefineLB.a RefineLB.o 
	
CommLB.decl.h: CommLB.ci charmxi
	$(CHARMC) CommLB.ci

CommLB.o: CommLB.C CommLB.decl.h manager.h $(CKHEADERS)
	$(CHARMC) -c CommLB.C 

$(L)/libmoduleCommLB.a: CommLB.o manager.o
	$(CHARMC) -o $(L)/libmoduleCommLB.a CommLB.o manager.o
	

Comm1LB.decl.h: Comm1LB.ci charmxi
	$(CHARMC) Comm1LB.ci

Comm1LB.o: Comm1LB.C Comm1LB.decl.h $(CKHEADERS)
	$(CHARMC) -c Comm1LB.C

$(L)/libmoduleComm1LB.a: Comm1LB.o 
	$(CHARMC) -o $(L)/libmoduleComm1LB.a Comm1LB.o 
	

GreedyLB.decl.h: GreedyLB.ci charmxi
	$(CHARMC) GreedyLB.ci

GreedyLB.o: GreedyLB.C GreedyLB.decl.h $(CKHEADERS)
	$(CHARMC) -c GreedyLB.C

$(L)/libmoduleGreedyLB.a: GreedyLB.o 
	$(CHARMC) -o $(L)/libmoduleGreedyLB.a GreedyLB.o 
	

NeighborLB.decl.h: NeighborLB.ci charmxi
	$(CHARMC) NeighborLB.ci

NeighborLB.o: NeighborLB.C NeighborLB.decl.h $(CKHEADERS)
	$(CHARMC) -c NeighborLB.C

$(L)/libmoduleNeighborLB.a: NeighborLB.o 
	$(CHARMC) -o $(L)/libmoduleNeighborLB.a NeighborLB.o 
	

GreedyRefLB.decl.h: GreedyRefLB.ci charmxi
	$(CHARMC) GreedyRefLB.ci

GreedyRefLB.o: GreedyRefLB.C GreedyRefLB.decl.h $(CKHEADERS)
	$(CHARMC) -c GreedyRefLB.C

$(L)/libmoduleGreedyRefLB.a: GreedyRefLB.o 
	$(CHARMC) -o $(L)/libmoduleGreedyRefLB.a GreedyRefLB.o 
	

OrbLB.decl.h: OrbLB.ci charmxi
	$(CHARMC) OrbLB.ci

OrbLB.o: OrbLB.C OrbLB.decl.h $(CKHEADERS)
	$(CHARMC) -c OrbLB.C

$(L)/libmoduleOrbLB.a: OrbLB.o 
	$(CHARMC) -o $(L)/libmoduleOrbLB.a OrbLB.o 
	

RandRefLB.decl.h: RandRefLB.ci charmxi
	$(CHARMC) RandRefLB.ci

RandRefLB.o: RandRefLB.C RandRefLB.decl.h $(CKHEADERS)
	$(CHARMC) -c RandRefLB.C

$(L)/libmoduleRandRefLB.a: RandRefLB.o 
	$(CHARMC) -o $(L)/libmoduleRandRefLB.a RandRefLB.o 
	

WSLB.decl.h: WSLB.ci charmxi
	$(CHARMC) WSLB.ci

WSLB.o: WSLB.C WSLB.decl.h $(CKHEADERS)
	$(CHARMC) -c WSLB.C

$(L)/libmoduleWSLB.a: WSLB.o 
	$(CHARMC) -o $(L)/libmoduleWSLB.a WSLB.o 
	

LB_OBJ=EveryLB.o \
    DummyLB.o \
    RandCentLB.o \
    RecBisectBfLB.o \
    MetisLB.o \
    RefineLB.o \
    CommLB.o \
    Comm1LB.o \
    GreedyLB.o \
    NeighborLB.o \
    GreedyRefLB.o \
    OrbLB.o \
    RandRefLB.o \
    WSLB.o \
    manager.o

EveryLB.o: EveryLB.C EveryLB.decl.h $(CVHEADERS)
	$(CHARMC) -c EveryLB.C

EveryLB.decl.h: EveryLB.ci $(CVHEADERS)
	$(CHARMC) EveryLB.ci

$(L)/libmoduleEveryLB.a: $(LB_OBJ)
	$(CHARMC) -o $(L)/libmoduleEveryLB.a $(LB_OBJ)
	cp libmoduleEveryLB.dep $(L)/
# end of from script Makefile_lb.sh
loadbalancers: $(LOADBALANCERS) $(L)/libmoduleEveryLB.a

###############################################################################
#
# BlueGene Emulator
#
###############################################################################

bluegene: charm++ $(L)/libconv-bluegene.a $(L)/libconv-bluegene-logs.a METIS

# old (Charm++) Blue Gene emulator
BlueGene.decl.h BlueGene.def.h : BlueGene.ci charmxi
	$(CHARMC) BlueGene.ci

$(L)/libbluegene.a: BlueGene.C BlueGene.h BlueGene.decl.h BlueGene.def.h
	$(CHARMC) -c BlueGene.C
	$(CHARMC) -o $@ BlueGene.o

# (Converse) Blue Gene emulator
BLUE_HEADERS = blue_types.h blue_timing.h blue_defs.h blue_logs.h blue.h blue_impl.h
BLUE_OBJS = blue.o blue_node.o blue_proc.o blue_init.o blue_timing.o
BLUE_IO_OBJS = blue_IO.o blue_logs.o

blue.o:  blue.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue.C

blue_node.o:  blue_node.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_node.C

blue_proc.o:  blue_proc.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_proc.C

blue_timing.o:  blue_timing.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_timing.C

blue_init.o:  blue_init.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_init.C

blue_logs.o:  blue_logs.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_logs.C

blue_IO.o:  blue_IO.C $(BLUE_HEADERS) $(CVHEADERS)
	$(CHARMC) -c blue_IO.C

$(L)/libconv-bluegene.a: $(BLUE_OBJS)
	$(CHARMC) -o $@  $(BLUE_OBJS)

$(L)/libconv-bluegene-logs.a: $(BLUE_IO_OBJS)
	$(CHARMC) -o $@  $(BLUE_IO_OBJS)

###############################################################################
#
# The CPM scanner
#
###############################################################################

conv-cpm: conv-cpm.o
	$(SEQCHARMC) -o conv-cpm -cp ../bin/ conv-cpm.o

conv-cpm.o: conv-cpm.c $(CVHEADERS)
	$(SEQCHARMC) conv-cpm.c

###############################################################################
#
# The interface translator
#
###############################################################################

XIHEADERS=conv-autoconfig.h xi-symbol.h xi-util.h xi-grammar.tab.h
CHARMXI=xi-main.o xi-symbol.o xi-grammar.tab.o xi-scan.o xi-util.o
SDAGOBJ=sdag-globals.o CSdagConstruct.o CEntry.o
SDAGHEADERS= CToken.h EToken.h CParsedFile.h \
	sdag-globals.h CEntry.h 

charmxi: $(CHARMXI) $(SDAGOBJ)
	$(SEQCHARMC) -language c++ -cp ../bin/ -o charmxi $(CHARMXI) $(SDAGOBJ)

xi-main.o: xi-main.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) xi-main.C

xi-symbol.o: xi-symbol.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) xi-symbol.C

xi-util.o: xi-util.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) xi-util.C

xi-grammar.tab.o: xi-grammar.tab.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) xi-grammar.tab.C

xi-scan.o: xi-scan.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) -I. -DYY_NEVER_INTERACTIVE xi-scan.C

sdag-globals.o: sdag-globals.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) -c sdag-globals.C

CSdagConstruct.o: CSdagConstruct.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) -c CSdagConstruct.C

CEntry.o: CEntry.C $(XIHEADERS) $(SDAGHEADERS)
	$(SEQCHARMC) -c CEntry.C

###############################################################################
#
# Simple Messaging
#
###############################################################################

$(L)/libtsm.a: converse simplemsg.c simplemsg.h
	$(CHARMC) -o $@ simplemsg.c

$(L)/libsm.a: converse sm.c sm.h
	$(CHARMC) -o $@ sm.c

###############################################################################
#
# PVM
#
###############################################################################

PVMC=pvmc_buf.o pvmc_comm.o pvmc_conv.o pvmc_main.o pvmc_pack.o pvmc_groups.o

$(L)/libpvmc.a: $(PVMC)
	$(CHARMC) -o $@ $(PVMC)

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

$(L)/libf90charm.a: f90main.ci f90main.h f90main.C charmf.h charmf.C futil.f90
	$(CHARMC) f90main.ci
	$(CHARMC) -c -DAMPI_FORTRAN=1 f90main.C
	$(CHARMC) -c -DAMPI_FORTRAN=1 charmf.C
	-$(CHARMC) -c futil.f90
	-$(CHARMC) -o $@ f90main.o charmf.o futil.o

###############################################################################
#
# Make clean
#
###############################################################################

install:
	-test ! -d $(DESTDIR)/include && mkdir -p $(DESTDIR)/include
	-test ! -d $(DESTDIR)/lib && mkdir -p $(DESTDIR)/lib
	-test ! -d $(DESTDIR)/lib_so && mkdir -p $(DESTDIR)/lib_so
	-test ! -d $(DESTDIR)/bin && mkdir -p $(DESTDIR)/bin
	cp -f ../include/*   $(DESTDIR)/include
	cp -f ../lib/*       $(DESTDIR)/lib
	cp -f ../lib_so/*    $(DESTDIR)/lib_so
	cp -f ../bin/*       $(DESTDIR)/bin

clean:
	rm -f conv-autoconfig.h config.cache
	rm -f QuickThreads/libqt.a
	rm -f charmxi conv-cpm
	rm -f TAGS basics cmk_extras core
	rm -f core *.a
	rm -f core *.o
	rm -f core #*#
	rm -f core *~
	if [ -d charmrun ] ; then ( cd charmrun; make clean ) ; fi
	-( cd doc && make clean )
	( cd libs; make clean )
	( cd ../pgms; make clean )

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


configure: configure.in
	-autoconf && autoheader
