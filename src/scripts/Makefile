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

charm++: basics
	$(MAKE) charmxi converse $(L)/libck.a loadbalancers default_libs

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

AMPI:  charm++
	$(MAKE) CONVLIBS 
	$(MAKE) AMPIC AMPIF

AMPIC: charm++ LIBS

AMPIF: AMPIC

f90charm: charm++ $(L)/libf90charm.a

default_libs: commlib multicast

multicast:
	cd libs/ck-libs/multicast && $(MAKE)

commlib: 
	cd libs/ck-libs/commlib && $(MAKE)

translators: charmxi conv-cpm

pose:  charm++
	cd libs/ck-libs/pose && $(MAKE) 

LIBS: CONVLIBS CHARMLIBS
	cd libs; $(MAKE) otherlibs 

test-all: test

test: test-charm test-AMPI test-fem
	@echo "--- All tests passed ---"

test-converse: converse
	cd ../pgms ; $(MAKE) test-converse  OPTS='$(OPTS)'

test-charm: charm++
	cd ../pgms ; $(MAKE) test-charm  OPTS='$(OPTS)'

test-AMPI: AMPI
	cd ../pgms ; $(MAKE) test-AMPI  OPTS='$(OPTS)'

test-fem: FEM
	cd ../pgms ; $(MAKE) test-fem  OPTS='$(OPTS)'

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

DEPENDFILE = Make.depends

include $(DEPENDFILE)

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
      tempo.decl.h waitqd.decl.h \
      CkFutures.decl.h CkCheckpoint.decl.h \
      LBDatabase.decl.h CentralLB.decl.h ComboCentLB.decl.h NullLB.decl.h \
      RandCentLB.decl.h RecBisectBfLB.decl.h BaseLB.decl.h \
      MetisLB.decl.h DummyLB.decl.h RefineLB.decl.h RefineCommLB.decl.h	\
      OrbLB.decl.h GreedyLB.decl.h NborBaseLB.decl.h NeighborLB.decl.h \
      WSLB.decl.h GreedyRefLB.decl.h RandRefLB.decl.h \
      GreedyCommLB.decl.h Comm1LB.decl.h EveryLB.decl.h \
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
	  Refiner.h RefinerComm.h ObjGraph.h heap.h elements.h CommLBHeap.h \
	  BaseLB.h CentralLB.h CentralLBMsg.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h RefineCommLB.h OrbLB.h GreedyLB.h GreedyCommLB.h \
	  Comm1LB.h manager.h MetisLB.h NborBaseLB.h \
	  NeighborLB.h NeighborLBMsg.h WSLB.h GreedyRefLB.h RandRefLB.h \
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
	$(MAKE) QuickThreads/libqt.a
	$(MAKE) converse-target

converse-target:  $(CVLIBS) fmain-ok charmrun-target swapglobal-target conv-cpm

charmrun-target: sockRoutines.c sockRoutines.h ccs-server.c ccs-server.h sockRoutines-seq.o
	if [ -d charmrun ] ; then ( cd charmrun ; $(MAKE) OPTS='$(OPTS)' ) ; fi
	if [ -f charmrun ] ; then ( cp charmrun ../bin ) ; fi

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
	  test -d ../lib_so && \
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

ccs-client.o: ccs-client.c $(CVHEADERS)
	$(CHARMC) ccs-client.c

sockRoutines.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -DCMK_USE_CONVERSE=1 sockRoutines.c

sockRoutines-seq.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -seq -o sockRoutines-seq.o sockRoutines.c

pup_util.o: pup_util.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_util.o pup_util.C

pup_xlater.o: pup_xlater.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_xlater.o pup_xlater.C

pup_c.o: pup_c.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_c.o pup_c.C

pup_paged.o: pup_paged.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_paged.o pup_paged.C
	
pup_f.o: pup_f.f90 $(CVHEADERS)
	-$(CHARMC) -c pup_f.f90 && $(CHARMC) -cpmod ../include pupmod.M

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
	   BaseLB.o CentralLB.o NborBaseLB.o WSLB.o \
           ObjGraph.o graph.o  \
           Refiner.o RefinerComm.o bitvecset.o Set.o heap.o CommLBHeap.o \
	   NullLB.o LBSimulation.o 	\
	   ComlibStrategy.o ComlibArrayListener.o  \
	   charmProjections.o

CHARMLIBS: charm++
	cd libs; $(MAKE) charmlibs OPTS='$(OPTS)'

$(L)/libck.a: $(LIBCK_CORE)
	$(CHARMC) -o $@ $(LIBCK_CORE)

CkFutures.def.h: CkFutures.decl.h

CkFutures.decl.h: ckfutures.ci charmxi
	$(CHARMC) ckfutures.ci

tempo.def.h: tempo.decl.h

tempo.decl.h: tempo.ci charmxi
	$(CHARMC) tempo.ci

waitqd.def.h: waitqd.decl.h

waitqd.decl.h: waitqd.ci charmxi
	$(CHARMC) waitqd.ci

LBDatabase.def.h: LBDatabase.decl.h

LBDatabase.decl.h: LBDatabase.ci charmxi
	$(CHARMC) LBDatabase.ci

NullLB.def.h: NullLB.decl.h

NullLB.decl.h: NullLB.ci charmxi
	$(CHARMC) NullLB.ci

BaseLB.def.h: BaseLB.decl.h

BaseLB.decl.h: BaseLB.ci charmxi
	$(CHARMC) BaseLB.ci

TraceSummary.def.h: TraceSummary.decl.h

TraceSummary.decl.h: trace-summary.ci charmxi
	$(CHARMC) trace-summary.ci

CkReduction.def.h: CkReduction.decl.h

CkReduction.decl.h: ckreduction.ci charmxi
	$(CHARMC) ckreduction.ci

CkCheckpoint.def.h: CkCheckpoint.decl.h

CkCheckpoint.decl.h: ckcheckpoint.ci charmxi
	$(CHARMC) ckcheckpoint.ci

CkMarshall.def.h: CkMarshall.decl.h

CkMarshall.decl.h: ckmarshall.ci charmxi
	$(CHARMC) ckmarshall.ci

CkArrayReductionMgr.def.h: CkArrayReductionMgr.decl.h

CkArrayReductionMgr.decl.h: ckarrayreductionmgr.ci charmxi
	$(CHARMC) ckarrayreductionmgr.ci

CkArray.def.h: CkArray.decl.h

CkArray.decl.h : ckarray.ci charmxi
	$(CHARMC) ckarray.ci

CkLocation.def.h: CkLocation.decl.h

CkLocation.decl.h: cklocation.ci charmxi
	$(CHARMC) cklocation.ci

CkCallback.def.h: CkCallback.decl.h

CkCallback.decl.h: ckcallback.ci charmxi
	$(CHARMC) ckcallback.ci

charisma.def.h: charisma.decl.h

charisma.decl.h: charisma.ci charmxi
	$(CHARMC) charisma.ci

## Charm++ Load balancer (Dynamic load balancer)

CentralLB.def.h: CentralLB.decl.h

CentralLB.decl.h: CentralLB.ci charmxi
	$(CHARMC) CentralLB.ci

NborBaseLB.def.h: NborBaseLB.decl.h

NborBaseLB.decl.h: NborBaseLB.ci charmxi
	$(CHARMC) NborBaseLB.ci


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

# used for make depends
TRACE_OBJS =  trace-projections.o  trace-summary.o  trace-counter.o	\
	      trace-bluegene.o trace-projector.o trace-converse.o trace-all.o

###############################################################################
#
# Dynamic Load Balancers
#
###############################################################################

# Automatically generated by script Makefile_lb.sh
#  on Mon Oct 27 12:01:47 CST 2003
LOADBALANCERS=\
   $(L)/libmoduleDummyLB.a \
   $(L)/libmoduleComboCentLB.a \
   $(L)/libmoduleRandCentLB.a \
   $(L)/libmoduleRecBisectBfLB.a \
   $(L)/libmoduleMetisLB.a \
   $(L)/libmoduleRefineLB.a \
   $(L)/libmoduleRefineCommLB.a \
   $(L)/libmoduleGreedyCommLB.a \
   $(L)/libmoduleComm1LB.a \
   $(L)/libmoduleGreedyLB.a \
   $(L)/libmoduleNeighborLB.a \
   $(L)/libmoduleGreedyRefLB.a \
   $(L)/libmoduleOrbLB.a \
   $(L)/libmoduleRandRefLB.a \
   $(L)/libmoduleWSLB.a \
   manager.o

DummyLB.def.h: DummyLB.decl.h

DummyLB.decl.h: DummyLB.ci charmxi
	$(CHARMC) DummyLB.ci

$(L)/libmoduleDummyLB.a: DummyLB.o 
	$(CHARMC) -o $(L)/libmoduleDummyLB.a DummyLB.o 
	

ComboCentLB.def.h: ComboCentLB.decl.h

ComboCentLB.decl.h: ComboCentLB.ci charmxi
	$(CHARMC) ComboCentLB.ci

$(L)/libmoduleComboCentLB.a: ComboCentLB.o 
	$(CHARMC) -o $(L)/libmoduleComboCentLB.a ComboCentLB.o 
	

RandCentLB.def.h: RandCentLB.decl.h

RandCentLB.decl.h: RandCentLB.ci charmxi
	$(CHARMC) RandCentLB.ci

$(L)/libmoduleRandCentLB.a: RandCentLB.o 
	$(CHARMC) -o $(L)/libmoduleRandCentLB.a RandCentLB.o 
	

RecBisectBfLB.def.h: RecBisectBfLB.decl.h

RecBisectBfLB.decl.h: RecBisectBfLB.ci charmxi
	$(CHARMC) RecBisectBfLB.ci

$(L)/libmoduleRecBisectBfLB.a: RecBisectBfLB.o 
	$(CHARMC) -o $(L)/libmoduleRecBisectBfLB.a RecBisectBfLB.o 
	

MetisLB.def.h: MetisLB.decl.h

MetisLB.decl.h: MetisLB.ci charmxi
	$(CHARMC) MetisLB.ci

$(L)/libmoduleMetisLB.a: MetisLB.o 
	$(CHARMC) -o $(L)/libmoduleMetisLB.a MetisLB.o 
	cp libmoduleMetisLB.dep $(L)/

RefineLB.def.h: RefineLB.decl.h

RefineLB.decl.h: RefineLB.ci charmxi
	$(CHARMC) RefineLB.ci

$(L)/libmoduleRefineLB.a: RefineLB.o 
	$(CHARMC) -o $(L)/libmoduleRefineLB.a RefineLB.o 
	

RefineCommLB.def.h: RefineCommLB.decl.h

RefineCommLB.decl.h: RefineCommLB.ci charmxi
	$(CHARMC) RefineCommLB.ci

$(L)/libmoduleRefineCommLB.a: RefineCommLB.o 
	$(CHARMC) -o $(L)/libmoduleRefineCommLB.a RefineCommLB.o 
	

GreedyCommLB.def.h: GreedyCommLB.decl.h

GreedyCommLB.decl.h: GreedyCommLB.ci charmxi
	$(CHARMC) GreedyCommLB.ci

$(L)/libmoduleGreedyCommLB.a: GreedyCommLB.o manager.o
	$(CHARMC) -o $(L)/libmoduleGreedyCommLB.a GreedyCommLB.o manager.o
	

Comm1LB.def.h: Comm1LB.decl.h

Comm1LB.decl.h: Comm1LB.ci charmxi
	$(CHARMC) Comm1LB.ci

$(L)/libmoduleComm1LB.a: Comm1LB.o 
	$(CHARMC) -o $(L)/libmoduleComm1LB.a Comm1LB.o 
	

GreedyLB.def.h: GreedyLB.decl.h

GreedyLB.decl.h: GreedyLB.ci charmxi
	$(CHARMC) GreedyLB.ci

$(L)/libmoduleGreedyLB.a: GreedyLB.o 
	$(CHARMC) -o $(L)/libmoduleGreedyLB.a GreedyLB.o 
	

NeighborLB.def.h: NeighborLB.decl.h

NeighborLB.decl.h: NeighborLB.ci charmxi
	$(CHARMC) NeighborLB.ci

$(L)/libmoduleNeighborLB.a: NeighborLB.o 
	$(CHARMC) -o $(L)/libmoduleNeighborLB.a NeighborLB.o 
	

GreedyRefLB.def.h: GreedyRefLB.decl.h

GreedyRefLB.decl.h: GreedyRefLB.ci charmxi
	$(CHARMC) GreedyRefLB.ci

$(L)/libmoduleGreedyRefLB.a: GreedyRefLB.o 
	$(CHARMC) -o $(L)/libmoduleGreedyRefLB.a GreedyRefLB.o 
	

OrbLB.def.h: OrbLB.decl.h

OrbLB.decl.h: OrbLB.ci charmxi
	$(CHARMC) OrbLB.ci

$(L)/libmoduleOrbLB.a: OrbLB.o 
	$(CHARMC) -o $(L)/libmoduleOrbLB.a OrbLB.o 
	

RandRefLB.def.h: RandRefLB.decl.h

RandRefLB.decl.h: RandRefLB.ci charmxi
	$(CHARMC) RandRefLB.ci

$(L)/libmoduleRandRefLB.a: RandRefLB.o 
	$(CHARMC) -o $(L)/libmoduleRandRefLB.a RandRefLB.o 
	

WSLB.def.h: WSLB.decl.h

WSLB.decl.h: WSLB.ci charmxi
	$(CHARMC) WSLB.ci

$(L)/libmoduleWSLB.a: WSLB.o 
	$(CHARMC) -o $(L)/libmoduleWSLB.a WSLB.o 
	

# used for make dependes
LB_OBJ=EveryLB.o \
    DummyLB.o \
    ComboCentLB.o \
    RandCentLB.o \
    RecBisectBfLB.o \
    MetisLB.o \
    RefineLB.o \
    RefineCommLB.o \
    GreedyCommLB.o \
    Comm1LB.o \
    GreedyLB.o \
    NeighborLB.o \
    GreedyRefLB.o \
    OrbLB.o \
    RandRefLB.o \
    WSLB.o \
    manager.o

EveryLB.def.h: EveryLB.decl.h

EveryLB.decl.h: EveryLB.ci charmxi
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

bluegene:  basics
	$(MAKE) bluegene-target

bluegene-target: charm++ $(L)/libconv-bluegene.a $(L)/libconv-bluegene-logs.a METIS

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
	mv Make.depends .Make.depends
	rm -rf [a-z]* [A-Z]*
	mv .Make.depends Make.depends
	rm -rf .[a-z]* .[A-Z]*


configure: configure.in
	-autoconf && autoheader

depends: basics 
	echo "Creating " $(DEPENDFILE) " ..."; 
	if [ -f $(DEPENDFILE) ]; then \
           /bin/cp -f $(DEPENDFILE) $(DEPENDFILE).old; \
        fi; \
	echo '#generated by make depends' > $(DEPENDFILE); \
	processFiles=''; 	\
        for i in $(LIBCONV_CORE) $(TRACE_OBJS) $(LIBCK_CORE) $(LB_OBJ) $(BLUE_OBJS) $(BLUE_IO_OBJS) ; do \
	      processed='false';	\
	      for f in $$processFiles; do 	\
		if test $$i = $$f ; then	\
		  processed=true;  break;	\
		fi;	\
	      done;	\
	      test $$processed = 'true'  && continue;	\
	      processFiles="$$processFiles $$i";	\
	      SRCFILE=`basename $$i .o`.C ; \
	      [ ! -f $$SRCFILE ] && SRCFILE=`basename $$i .o`.c ;	\
              echo "checking dependencies for $$SRCFILE" ; \
              g++ -MM -Wno-deprecated -I. -I$(INC) $$SRCFILE  >> $(DEPENDFILE); \
              echo '	$$(CHARMC) -c -I. '$$SRCFILE >> $(DEPENDFILE) ; \
	      echo '' >> $(DEPENDFILE) ; \
        done; 

