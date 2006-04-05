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
	$(MAKE) charm-target
	@echo "$(OPTS)" > OPTS

bgampi: bluegene AMPI

TSM: converse $(L)/libtsm.a

SM: converse $(L)/libsm.a

PVM: converse $(L)/libpvmc.a

AMR: charm++ LIBS
	cd libs/ck-libs/amr; make all OPTS='$(OPTS)'

FEM: charm++
	$(MAKE) CONVLIBS
	$(MAKE) FEMC LIBS FEMF

FEMC: charm++ LIBS

FEMF: FEMC

ParFUM: FEM

MBLOCK: AMPI MBLOCKC MBLOCKF

MBLOCKC: charm++ LIBS

MBLOCKF: MBLOCKC

AMPI:  charm++
	$(MAKE) CONVLIBS
	$(MAKE) AMPIC AMPIF

ARMCI: charm++ LIBS

AMPIC: charm++ LIBS

AMPIF: AMPIC

f90charm: charm++ $(L)/libf90charm.a

default_libs: multicast

multicast:
	cd libs/ck-libs/multicast && $(MAKE)

translators: charmxi conv-cpm

pose:  charm++
	cd libs/ck-libs/pose && $(MAKE)

msa: charm++
	cd libs/ck-libs/multiphaseSharedArrays && $(MAKE)

jade: charm++ msa
#	make the lib, include
	cd langs/jade && $(MAKE)

fftlib: charm++
	cd libs/ck-libs/fftlib && $(MAKE)

# This should be done offline, since the user does not need to compile
# parser files, etc.  All they need is the libjade + include files,
# antlr.jar, jade.jar, and a few scripts which the jade target takes
# care of.
jadeall: jade
#	make parser.g -> .java files
	cd langs/jade && $(MAKE) p
#	make the jade.jar file
	cd ../java/charm/jade && $(MAKE)

LIBS: CONVLIBS CHARMLIBS
	cd libs; $(MAKE) otherlibs

test-all: test

bgtest: bluegene AMPI
	cd ../tests ; $(MAKE) bgtest OPTS='$(OPTS)'
	cd ../examples ; $(MAKE) bgtest OPTS='$(OPTS)'

test: bluegene AMPI
	cd ../tests ; $(MAKE) test OPTS='$(OPTS)'
	cd ../examples ; $(MAKE) test OPTS='$(OPTS)'

test-converse: converse
	cd ../tests ; $(MAKE) test-converse  OPTS='$(OPTS)'

test-charm: charm++ METIS
	cd ../tests ; $(MAKE) test-charm  OPTS='$(OPTS)'

test-AMPI: AMPI
	cd ../tests ; $(MAKE) test-AMPI  OPTS='$(OPTS)'

test-fem: FEM
	cd ../tests ; $(MAKE) test-fem  OPTS='$(OPTS)'

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

NATIVECHARMC=../bin/charmc -host $(CHARMOPTS)

CHARMXI=../bin/charmc -intrinsic $(OPTS)

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
      conv-ccs.h ccs-client.c ccs-client.h \
      ccs-server.h ccs-auth.c ccs-auth.h \
      memory-isomalloc.h debug-conv.h debug-charm.h conv-autoconfig.h \
      conv-common.h conv-config.sh conv-config.h conv-mach.h conv-mach.sh \
      blue.h blue-conv.h bgconverse.h \
	convcomlibstrategy.h 3dgridrouter.h de.h hypercubetopology.h	\
	routerstrategy.h convcomlib.h dummyconversestrategy.h  treerouter.h \
	convcomlibmanager.h graphrouter.h petable.h gridrouter.h router.h \
	prefixrouter.h pipebroadcastconverse.h pipelinestrategy.h cmipool.h \
	comlib.h

# The .c files are there to be #included by clients whole
# This is a bit unusual, but makes client linking simpler.
UTILHEADERS=pup.h pupf.h pup_c.h pup_stl.h pup_mpi.h pup_toNetwork.h pup_toNetwork4.h pup_paged.h pup_cmialloc.h\
	ckimage.h ckdll.h ckhashtable.h ckbitvector.h cklists.h ckliststring.h \
	ckstatistics.h ckvector3d.h conv-lists.h RTH.h ckcomplex.h \
	sockRoutines.h sockRoutines.c ckpool.h bgltorus.h

CKHEADERS=ck.h ckstream.h envelope.h init.h qd.h charm.h charm++.h \
	  ckfutures.h \
	  ckcallback.h CkCallback.decl.h ckcallback-ccs.h 	\
	  ckarrayreductionmgr.h cknodegroupreduction.h cksection.h \
	  ckarray.h cklocation.h ckreduction.h ckcheckpoint.h ckmemcheckpoint.h ckevacuation.h\
	  ckobjQ.h readonly.h charisma.h ComlibArrayListener.h ComlibStrategy.h\
	  ComlibLearner.h $(UTILHEADERS) \
	  tempo.h waitqd.h LBDatabase.h lbdb.h LBProfit.h LBDBManager.h	\
	  LBComm.h LBOM.h LBObj.h LBMachineUtil.h LBAgent.h \
	  Refiner.h RefinerApprox.h RefinerComm.h ObjGraph.h heap.h \
          elements.h CommLBHeap.h topology.h manager.h \
	  BaseLB.h CentralLB.h CentralLBMsg.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h RefineKLB.h RefineCommLB.h OrbLB.h \
	  GreedyLB.h GreedyCommLB.h GreedyAgentLB.h Comm1LB.h MetisLB.h \
	  TopoLB.h RefineTopoLB.h  TopoCentLB.h  PhasebyArrayLB.h RotateLB.h \
	  GridCommLB.h GridCommRefineLB.h DummyLB.h BlockLB.h \
	  NborBaseLB.h HybridBaseLB.h HybridLB.h HybridLBMsg.h HbmLB.h \
	  NeighborLB.h NeighborCommLB.h NeighborLBMsg.h WSLB.h \
	  BlueGene.h middle.h middle-conv.h middle-blue.h \
          CkMarshall.decl.h CkArray.decl.h CkLocation.decl.h 	\
          CkReduction.decl.h CkArrayReductionMgr.decl.h\
          tempo.decl.h waitqd.decl.h \
          CkFutures.decl.h CkCheckpoint.decl.h CkMemCheckpoint.decl.h \
          LBDatabase.decl.h CentralLB.decl.h ComboCentLB.decl.h NullLB.decl.h \
          RandCentLB.decl.h RecBisectBfLB.decl.h BaseLB.decl.h MetisLB.decl.h \
          DummyLB.decl.h RotateLB.decl.h RefineLB.decl.h RefineKLB.decl.h BlockLB.decl.h \
          RefineCommLB.decl.h OrbLB.decl.h GreedyLB.decl.h GreedyCommLB.decl.h \
          Comm1LB.decl.h GreedyAgentLB.decl.h GridCommLB.decl.h GridCommRefineLB.decl.h \
	  PhasebyArrayLB.decl.h TopoLB.decl.h RefineTopoLB.decl.h 	\
	  TopoCentLB.decl.h NborBaseLB.decl.h NeighborLB.decl.h \
          NeighborCommLB.decl.h HbmLB.decl.h \
          HybridBaseLB.decl.h HybridLB.decl.h WSLB.decl.h EveryLB.decl.h \
          charisma.decl.h TraceSummary.decl.h BlueGene.decl.h \
	  comlib.decl.h ComlibManager.h DummyStrategy.h	\
	  DirectMulticastStrategy.h RingMulticastStrategy.h	\
	  MultiRingMulticast.h \
	  EachToManyMulticastStrategy.h MsgPacker.h StreamingStrategy.h \
	  NodeMulticast.h PipeBroadcastStrategy.h MeshStreamingStrategy.h \
          PrioStreaming.h MPIStrategy.h MsgPacker.h StreamingStrategy.h \
	  BroadcastStrategy.h ComlibSectionInfo.h AAPLearner.h \
	  AAMLearner.h ComlibStats.h \
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
	rm -rf ../lib_so ; mkdir ../lib_so; touch ../lib_so/.charmso
	rm -rf ../examples
	rm -rf ../tests
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
	./gatherflat ../../src/ck-com           .
	./gatherflat ../../src/conv-com           .
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
	./gathertree ../../src/langs/jade langs/jade
	./gathertree ../../src/arch/common           .
	./gathertree ../../src/arch/`cat .gdir`      .
	./gatherflat ../../src/arch/`cat .vdir`      .
	./gathertree ../../examples ../examples
	./gathertree ../../tests ../tests
	rm -f ../bin/dep.pl ; cp dep.pl ../bin/
	chmod +x charmc
	-./system_ln  ../tmp/charmc ../bin/
	-ln -s ../../java/bin charmjavabin
	-ln -s ../../java/bin ../bin/charmjavabin
	-ln -s charmjavabin/* ../bin/
	-ln -s ../java ..
	rm -f .gdir
	rm -rf ../include ; mkdir ../include
	-./system_ln  ../tmp/conv-*.*h ../include
	-./system_ln  ../tmp/cc-*.*h ../include
	if [ ! -f conv-common.h ] ; then ( touch conv-common.h ) ; fi
	if [ -x ./special.sh ] ; then ./special.sh ; fi
	touch dirs+sources

###############################################################################
#
# Converse Libraries
#
###############################################################################
CLBLIBS=$(L)/libldb-rand.o $(L)/libldb-spray.o $(L)/libldb-neighbor.o $(L)/libldb-none.o $(L)/libldb-test.o $(L)/libldb-bluegene.o
TRACELIBS=$(L)/libtrace-projections.a  $(L)/libtrace-summary.a \
          $(L)/libtrace-counter.a $(L)/libtrace-bluegene.a \
	  $(L)/libtrace-projector.a $(L)/libtrace-all.a $(L)/libtrace-converse.a \
          $(L)/libtrace-recordreplay.a  $(L)/libtracef_f.a
MEMLIBS=$(L)/libmemory-default.o $(L)/libmemory-os.o $(L)/libmemory-gnu.o \
          $(L)/libmemory-gnuold.o $(L)/libmemory-verbose.o 	\
	  $(L)/libmemory-paranoid.o \
	  $(L)/libmemory-leak.o $(L)/libmemory-isomalloc.o \
	  $(L)/libmemory-charmdebug.o
THREADLIBS=$(L)/libthreads-default.o $(L)/libthreads-qt.o 	\
	  $(L)/libthreads-context.o $(L)/libthreads-uJcontext.o \
	  $(L)/libthreads-pthreads.o $(L)/libthreads-fibers.o \
	  $(L)/libthreads-stackcopy.o $(L)/libthreads-memoryalias.o \
          $(L)/libthreads-qt-memoryalias.o	\
	  $(L)/libthreads-context-memoryalias.o	\
	  $(L)/libthreads-uJcontext-memoryalias.o

CVLIBS=$(L)/libconv-core.a \
	$(L)/libconv-cplus-y.a $(L)/libconv-cplus-n.a \
	$(L)/libconv-util.a $(L)/libconv-utilf.a \
	$(CLBLIBS) $(TRACELIBS) $(MEMLIBS) $(THREADLIBS) \
	$(L)/libccs-client.a

LIBCONV_CORE= convcore.o conv-conds.o queueing.o msgmgr.o \
	cpm.o cpthreads.o futures.o cldb.o topology.o random.o \
	debug-conv.o generate.o edgelist.o conv-ccs.o ccs-builtins.o \
	traceCore.o traceCoreCommon.o \
	converseProjections.o machineProjections.o \
	quiescence.o isomalloc.o conv-counter.o \
	global-nop.o cmipool.o

#############################################
#
#Comlib Core objects
#
#############################################
#These go into libconv-util.a and are always linked
CONVERSE_COM_CORE_OBJS = convcomlibstrategy.o

#These go into libck.a
COMLIB_CORE_OBJS = ComlibStrategy.o ComlibArrayListener.o
##############################################

LIBCONV_UTIL=pup_util.o pup_toNetwork.o pup_toNetwork4.o		\
	pup_xlater.o pup_c.o pup_paged.o pup_cmialloc.o			\
	ckimage.o ckdll.o ckhashtable.o sockRoutines.o			\
	conv-lists.o RTH.o persist-comm.o $(CONVERSE_COM_CORE_OBJS) \
	ckbitvector.o

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
	cd libs/ck-libs/parmetis/METISLib && $(MAKE) OPTS='$(OPTS)'

CONVLIBS: converse
	cd libs; $(MAKE) convlibs OPTS='$(OPTS)'

$(L)/libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) -o $@ $(LIBCONV_CORE)

$(L)/libconv-cplus-y.a: $(L)/libconv-cplus-n.a

include Makefile.machine

$(L)/libconv-cplus-n.a: machine.c $(CVHEADERS)
	@-test -f $(INC)/mpi.h && mv -f $(INC)/mpi.h $(INC)/mpi.BAK || true
	$(CHARMC) -o machine.o  -DFOR_CPLUS=1 machine.c
	@-test -f ./bglmachine.C && $(CHARMC) -o machine.o  -DFOR_CPLUS=1 bglmachine.C || true
	$(CHARMC) -o $(L)/libconv-cplus-y.a machine.o
	$(CHARMC) -o $(L)/libconv-cplus-n.a -DFOR_CPLUS=0 machine.c
	@-test -f $(INC)/mpi.BAK && mv -f $(INC)/mpi.BAK $(INC)/mpi.h || true

$(L)/libconv-util.a: $(LIBCONV_UTIL)
	$(CHARMC) -o $@ $(LIBCONV_UTIL)

$(L)/libconv-utilf.a: $(LIBCONV_UTILF)
	-$(CHARMC) -o $@ $(LIBCONV_UTILF) || touch $@

$(L)/libccs-client.a: ccs-client.o
	$(CHARMC) -o $@ ccs-client.o

ccs-client.o: ccs-client.c $(CVHEADERS)
	$(CHARMC) ccs-client.c

sockRoutines.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -DCMK_USE_CONVERSE=1 sockRoutines.c

sockRoutines-seq.o: sockRoutines.c $(CVHEADERS) $(UTILHEADERS)
	$(SEQCHARMC) -o sockRoutines-seq.o sockRoutines.c

pup_util.o: pup_util.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_util.o pup_util.C

pup_toNetwork.o: pup_toNetwork.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_toNetwork.o pup_toNetwork.C

pup_toNetwork4.o: pup_toNetwork4.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_toNetwork4.o pup_toNetwork4.C

pup_xlater.o: pup_xlater.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_xlater.o pup_xlater.C

pup_c.o: pup_c.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_c.o pup_c.C

pup_paged.o: pup_paged.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_paged.o pup_paged.C

pup_f.o: pup_f.f90 $(CVHEADERS)
	-$(CHARMC) -c pup_f.f90 && $(CHARMC) -cpmod ../include pupmod.M || touch pup_f.o

pup_cmialloc.o: pup_cmialloc.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o pup_cmialloc.o pup_cmialloc.C

ckhashtable.o: ckhashtable.C ckhashtable.h $(CVHEADERS)
	$(CHARMC) -o ckhashtable.o ckhashtable.C

ckbitvector.o: ckbitvector.C ckbitvector.h $(CVHEADERS)
	$(CHARMC) -o ckbitvector.o ckbitvector.C

ckdll.o: ckdll.C ckdll.h $(CVHEADERS)
	$(CHARMC) -o ckdll.o ckdll.C

ckimage.o: ckimage.C ckimage.h $(CVHEADERS)
	$(CHARMC) -o ckimage.o ckimage.C

RTH.o: RTH.C RTH.h $(CVHEADERS)
	$(CHARMC) -o RTH.o RTH.C

persist-comm.o: persist-comm.c persist_impl.h $(CVHEADERS)
	$(CHARMC) -o persist-comm.o persist-comm.c

conv-lists.o: conv-lists.C $(UTILHEADERS) $(CVHEADERS)
	$(CHARMC) -o conv-lists.o conv-lists.C

## Converse load balancers (seed balancers, -balance)
$(L)/libldb-none.o: cldb.none.c $(CVHEADERS)
	$(CHARMC) -o $@ cldb.none.c

$(L)/libldb-rand.o: cldb.rand.c cldb.h $(CVHEADERS)
	$(CHARMC) -o $@ cldb.rand.c

$(L)/libldb-neighbor.o: cldb.neighbor.c cldb.neighbor.h graph.h $(CVHEADERS)
	$(CHARMC) -o $@ cldb.neighbor.c

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
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_GNU memory.c || touch $@

$(L)/libmemory-gnuold.o: $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_GNUOLD memory.c || touch $@

$(L)/libmemory-verbose.o: memory-verbose.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_VERBOSE memory.c || touch $@

$(L)/libmemory-paranoid.o: memory-paranoid.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_PARANOID memory.c || touch $@

$(L)/libmemory-leak.o: memory-leak.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_LEAK memory.c || touch $@

$(L)/libmemory-cache.o: memory-cache.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_CACHE memory.c || touch $@

$(L)/libmemory-isomalloc.o: memory-isomalloc.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_ISOMALLOC memory.c || touch $@

$(L)/libmemory-charmdebug.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_CHARMDEBUG memory.c || touch $@

## Thread libraries (-thread)
$(L)/libthreads-default.o: threads.c $(CVHEADERS)
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_DEFAULT=1 -IQuickThreads -I. threads.c

$(L)/libthreads-qt.o: threads.c $(CVHEADERS)
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_QT=1 -IQuickThreads threads.c

$(L)/libthreads-context.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_CONTEXT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-uJcontext.o: threads.c uJcontext.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_JCONTEXT=1 -IQuickThreads -I. threads.c || touch $@

$(L)/libthreads-pthreads.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_PTHREADS=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-fibers.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_FIBERS=1 -IQuickThreads threads.c 2> /dev/null || touch $@

$(L)/libthreads-stackcopy.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_STACKCOPY=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-memoryalias.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_DEFAULT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-qt-memoryalias.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_QT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-context-memoryalias.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_CONTEXT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-uJcontext-memoryalias.o: threads.c $(CVHEADERS)
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_JCONTEXT=1 -IQuickThreads -I. threads.c || touch $@

## Global swapping (-swapglobal)
swapglobal-target: $(L)/libglobal-swap.o

$(L)/libglobal-swap.o: global-elfgot.C $(CVHEADERS)
	-$(CHARMC) -c global-elfgot.C -o $@ || touch $@


###############################################################################
#
# Charm Libraries
#
###############################################################################

CK_LIBS_CORE=$(L)/libck.a

LIBCK_CORE=trace-common.o tracef.o init.o register.o qd.o ck.o main.o  \
	   msgalloc.o ckfutures.o debug-message.o debug-charm.o ckcallback.o \
	   cklocation.o ckarray.o ckreduction.o ckarrayreductionmgr.o\
           tempo.o waitqd.o LBDatabase.o lbdb.o lbdbf.o charisma.o ckobjQ.o  \
	   LBAgent.o LBProfit.o ckcheckpoint.o ckmemcheckpoint.o ckevacuation.o\
           LBDBManager.o LBComm.o LBObj.o LBMachineUtil.o CentralPredictor.o \
	   BaseLB.o CentralLB.o HybridBaseLB.o NborBaseLB.o WSLB.o \
           ObjGraph.o graph.o LButil.o Refiner.o RefinerApprox.o  \
           RefinerComm.o bitvecset.o Set.o heap.o CommLBHeap.o \
	   NullLB.o LBSimulation.o $(COMLIB_CORE_OBJS) \
	   charmProjections.o

charm-target: converse $(L)/libck.a $(L)/libckf.a loadbalancers default_libs comlib_objs

CHARMLIBS: charm++
	cd libs; $(MAKE) charmlibs OPTS='$(OPTS)'

$(L)/libck.a: $(LIBCK_CORE)
	$(CHARMC) -o $@ $(LIBCK_CORE)

CkFutures.def.h: CkFutures.decl.h

CkFutures.decl.h: ckfutures.ci charmxi
	$(CHARMXI) ckfutures.ci

tempo.def.h: tempo.decl.h

tempo.decl.h: tempo.ci charmxi
	$(CHARMXI) tempo.ci

waitqd.def.h: waitqd.decl.h

waitqd.decl.h: waitqd.ci charmxi
	$(CHARMXI) waitqd.ci

LBDatabase.def.h: LBDatabase.decl.h

LBDatabase.decl.h: LBDatabase.ci charmxi
	$(CHARMXI) LBDatabase.ci

NullLB.def.h: NullLB.decl.h

NullLB.decl.h: NullLB.ci charmxi
	$(CHARMXI) NullLB.ci

BaseLB.def.h: BaseLB.decl.h

BaseLB.decl.h: BaseLB.ci charmxi
	$(CHARMXI) BaseLB.ci

TraceSummary.def.h: TraceSummary.decl.h

TraceSummary.decl.h: trace-summary.ci charmxi
	$(CHARMXI) trace-summary.ci

CkReduction.def.h: CkReduction.decl.h

CkReduction.decl.h: ckreduction.ci charmxi
	$(CHARMXI) ckreduction.ci

CkCheckpoint.def.h: CkCheckpoint.decl.h

CkCheckpoint.decl.h: ckcheckpoint.ci charmxi
	$(CHARMXI) ckcheckpoint.ci

CkMemCheckpoint.def.h: CkMemCheckpoint.decl.h

CkMemCheckpoint.decl.h: ckmemcheckpoint.ci charmxi
	$(CHARMXI) ckmemcheckpoint.ci

CkMarshall.def.h: CkMarshall.decl.h

CkMarshall.decl.h: ckmarshall.ci charmxi
	$(CHARMXI) ckmarshall.ci

CkArrayReductionMgr.def.h: CkArrayReductionMgr.decl.h

CkArrayReductionMgr.decl.h: ckarrayreductionmgr.ci charmxi
	$(CHARMXI) ckarrayreductionmgr.ci

CkArray.def.h: CkArray.decl.h

CkArray.decl.h : ckarray.ci charmxi
	$(CHARMXI) ckarray.ci

CkLocation.def.h: CkLocation.decl.h

CkLocation.decl.h: cklocation.ci charmxi
	$(CHARMXI) cklocation.ci

CkCallback.def.h: CkCallback.decl.h

CkCallback.decl.h: ckcallback.ci charmxi
	$(CHARMXI) ckcallback.ci

charisma.def.h: charisma.decl.h

charisma.decl.h: charisma.ci charmxi
	$(CHARMXI) charisma.ci

## Charm++ Load balancer (Dynamic load balancer)

CentralLB.def.h: CentralLB.decl.h

CentralLB.decl.h: CentralLB.ci charmxi
	$(CHARMXI) CentralLB.ci

NborBaseLB.def.h: NborBaseLB.decl.h

NborBaseLB.decl.h: NborBaseLB.ci charmxi
	$(CHARMXI) NborBaseLB.ci

HybridBaseLB.def.h: HybridBaseLB.decl.h

HybridBaseLB.decl.h: HybridBaseLB.ci charmxi
	$(CHARMXI) HybridBaseLB.ci

## Charm fortran AMPI

CK_FOR_LIBS_CORE=$(L)/libckf.a

LIBCKF_CORE=charmf.o 

$(L)/libckf.a: $(LIBCKF_CORE)
	$(CHARMC) -o $@ $(LIBCKF_CORE)

## Tracing libraries (profiling, -tracemode)
LIBTRACE_PROJ=trace-projections.o
$(L)/libtrace-projections.a: $(LIBTRACE_PROJ)
	$(CHARMC) -o $@ $(LIBTRACE_PROJ)

LIBTRACE_SUMM=trace-summary.o
$(L)/libtrace-summary.a: $(LIBTRACE_SUMM)
	$(CHARMC) -o $@ $(LIBTRACE_SUMM)

LIBTRACE_RECREP=trace-recordreplay.o
$(L)/libtrace-recordreplay.a: $(LIBTRACE_RECREP)
	$(CHARMC) -o $@ $(LIBTRACE_RECREP)

LIBTRACE_COUNT=trace-counter.o
$(L)/libtrace-counter.a: $(LIBTRACE_COUNT)
	$(CHARMC) -o $@ $(LIBTRACE_COUNT)

LIBTRACE_BG=trace-bluegene.o
$(L)/libtrace-bluegene.a:$(LIBTRACE_BG)
	$(CHARMC) -o $@ $(LIBTRACE_BG)

LIBTRACE_ALL=trace-all.o trace-projections.o trace-summary.o trace-recordreplay.o trace-projector.o traceCore.o traceCoreCommon.o charmProjections.o converseProjections.o machineProjections.o
$(L)/libtrace-all.a: $(LIBTRACE_ALL)
	$(CHARMC) -o $@ $(LIBTRACE_ALL)

LIBTRACE_PROJECTOR=trace-projector.o traceCore.o traceCoreCommon.o charmProjections.o converseProjections.o machineProjections.o
$(L)/libtrace-projector.a: $(LIBTRACE_PROJECTOR)
	$(CHARMC) -o  $@ $(LIBTRACE_PROJECTOR)

LIBTRACE_CONVERSE=trace-converse.o
$(L)/libtrace-converse.a: $(LIBTRACE_CONVERSE)
	$(CHARMC) -o  $@ $(LIBTRACE_CONVERSE)

LIBTRACE_FOR=tracef_f.o
$(L)/libtracef_f.a: $(LIBTRACE_FOR)
	-$(CHARMC) -o $@ $(LIBTRACE_FOR) || touch $@

tracef_f.o: tracef_f.f90
	-$(CHARMC) -o $@ -c tracef_f.f90 && $(CHARMC) -cpmod ../include tracemod.M  || touch $@

# used for make depends
TRACE_OBJS =  trace-projections.o  trace-summary.o  trace-counter.o	\
	      trace-bluegene.o trace-projector.o trace-converse.o trace-all.o \
              trace-recordreplay.o

###############################################################################
#
# Dynamic Load Balancers
#
###############################################################################


# Automatically generated by script Makefile_lb.sh
include Make.lb

loadbalancers: $(LOADBALANCERS) $(L)/libmoduleEveryLB.a


######################################################################
#
#Communication Optimization Library
#
######################################################################

STRATEGY_OBJS = MPIStrategy.o StreamingStrategy.o \
                DummyStrategy.o NodeMulticast.o \
		EachToManyMulticastStrategy.o ComlibSectionInfo.o \
                AAPLearner.o ComlibStats.o AAMLearner.o \
                MsgPacker.o DirectMulticastStrategy.o \
		RingMulticastStrategy.o MultiRingMulticast.o\
                PipeBroadcastStrategy.o MeshStreamingStrategy.o \
                PrioStreaming.o BroadcastStrategy.o

CONVERSE_COM_OBJS = gridrouter.o de.o treerouter.o petable.o \
	3dgridrouter.o graphrouter.o hypercubetopology.o \
	routerstrategy.o convcomlibmanager.o pipebroadcastconverse.o \
	pipelinestrategy.o prefixrouter.o

comlib.def.h: comlib.decl.h

comlib.decl.h : ComlibManager.ci charmxi
	$(CHARMXI) ComlibManager.ci

COMLIB_OBJS = ComlibManager.o $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)

comlib_objs: $(L)/libmodulecomlib.a $(L)/libconvcom.a

$(L)/libconvcom.a: $(COMLIB_OBJS)
	$(CHARMC) -o $(L)/libconvcom.a $(CONVERSE_COM_OBJS)

#for backward compatibility, keep libmodulecommlib.a with double m.
$(L)/libmodulecomlib.a: $(COMLIB_OBJS)
	$(CHARMC) -o $(L)/libmodulecomlib.a $(COMLIB_OBJS)
	$(CHARMC) -o $(L)/libmodulecommlib.a $(COMLIB_OBJS)

###############################################################################
#
# BlueGene Emulator
#
###############################################################################

bluegene:  basics
	$(MAKE) bluegene-target

bluegene-target: charm++ $(L)/libconv-bluegene.a $(L)/libconv-bluegene-logs.a $(L)/libconv-bluegene-stub.a $(L)/libblue-standalone.a METIS

# old (Charm++) Blue Gene emulator
BlueGene.decl.h BlueGene.def.h : BlueGene.ci charmxi
	$(CHARMXI) BlueGene.ci

$(L)/libbluegene.a: BlueGene.C BlueGene.h BlueGene.decl.h BlueGene.def.h
	$(CHARMC) -c BlueGene.C
	$(CHARMC) -o $@ BlueGene.o

# (Converse) Blue Gene emulator
BLUE_HEADERS = blue_types.h blue_timing.h blue_defs.h blue_logs.h blue.h blue_impl.h blue_network.h
BLUE_OBJS = blue.o blue_node.o blue_proc.o blue_init.o blue_timing.o
BLUE_IO_OBJS = blue_IO.o blue_logs.o blue_api.o

LIBBLUE_OBJS = $(BLUE_OBJS) $(BLUE_IO_OBJS) blue_stub.o blue_standalone.o

$(L)/libconv-bluegene.a: $(BLUE_OBJS)
	$(CHARMC) -o $@  $(BLUE_OBJS)

$(L)/libconv-bluegene-logs.a: $(BLUE_IO_OBJS)
	$(CHARMC) -o $@  $(BLUE_IO_OBJS)

$(L)/libconv-bluegene-stub.a:  blue_stub.o
	$(CHARMC) -o $@  blue_stub.o

$(L)/libblue-standalone.a: blue_standalone.o
	$(CHARMC) -o $@  blue_standalone.o

###############################################################################
#
# The CPM scanner
#
###############################################################################

conv-cpm: conv-cpm.o
	$(NATIVECHARMC) -language c -o conv-cpm -cp ../bin/ conv-cpm.o

conv-cpm.o: conv-cpm.c $(CVHEADERS)
	$(NATIVECHARMC) conv-cpm.c

###############################################################################
#
# The interface translator
#
###############################################################################

XIHEADERS=conv-autoconfig.h xi-symbol.h xi-util.h xi-grammar.tab.h
CHARMXIOBJ=xi-main.o xi-symbol.o xi-grammar.tab.o xi-scan.o xi-util.o
SDAGOBJ=sdag-globals.o CSdagConstruct.o CEntry.o
SDAGHEADERS= CToken.h EToken.h CParsedFile.h \
	sdag-globals.h CEntry.h

charmxi: ../bin/charmxi

../bin/charmxi: $(CHARMXIOBJ) $(SDAGOBJ)
	$(NATIVECHARMC) -language c++ -cp ../bin/ -o charmxi $(CHARMXIOBJ) $(SDAGOBJ)

xi-main.o: xi-main.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) xi-main.C

xi-symbol.o: xi-symbol.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) xi-symbol.C

xi-util.o: xi-util.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) xi-util.C

xi-grammar.tab.o: xi-grammar.tab.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) xi-grammar.tab.C

xi-scan.o: xi-scan.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) -I. -DYY_NEVER_INTERACTIVE xi-scan.C

sdag-globals.o: sdag-globals.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) -c sdag-globals.C

CSdagConstruct.o: CSdagConstruct.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) -c CSdagConstruct.C

CEntry.o: CEntry.C $(XIHEADERS) $(SDAGHEADERS)
	$(NATIVECHARMC) -c CEntry.C

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

$(L)/libf90charm.a: charmxi f90main.ci f90main.h f90main.C futil.f90
	$(CHARMC) f90main.ci
	$(CHARMC) -c -DAMPI_FORTRAN=1 f90main.C
	-$(CHARMC) -c futil.f90
	-$(CHARMC) -o $@ f90main.o futil.o

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
	( cd ../examples && make clean )
	( cd ../tests && make clean )

veryclean:
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../src
	rm -rf ../examples
	rm -rf ../tests
	rm -rf ../include
	rm -rf ../doc
	mv Makefile .Makefile
	mv Make.depends .Make.depends
	rm -rf [a-z]* [A-Z]*
	mv .Make.depends Make.depends
	rm -rf .[a-z]* .[A-Z]*


config: configure.in
	autoconf && autoheader

depends: basics
	echo "Creating " $(DEPENDFILE) " ...";
	if [ -f $(DEPENDFILE) ]; then \
           /bin/cp -f $(DEPENDFILE) $(DEPENDFILE).old; \
        fi; \
	echo '#generated by make depends' > $(DEPENDFILE); \
	processFiles=''; 	\
        for i in $(LIBCONV_CORE) $(TRACE_OBJS) $(LIBCK_CORE) $(LIBCKF_CORE) $(LB_OBJ) $(LIBBLUE_OBJS) $(CONVERSE_COM_CORE_OBJS) $(COMLIB_OBJS); do \
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
              if g++ -MM -Wno-deprecated -I. -I$(INC) $$SRCFILE  >> $(DEPENDFILE); then   echo '	$$(CHARMC) -c -I. '$$SRCFILE >> $(DEPENDFILE) ; else echo '' ; echo "Compilation of '$$SRCFILE' failed, please fix it first!!!!" ; exit; fi;  \
	      echo '' >> $(DEPENDFILE) ; \
        done;

