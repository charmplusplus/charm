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

all: charm++ bigsim LIBS AMPI f90charm TSM SM PVM doc

all-test: bigsim AMPI
	cd ../tests ; $(MAKE) all OPTS='$(OPTS)'
	cd ../examples ; $(MAKE) all OPTS='$(OPTS)'

charm++: charm-target
	@echo "$(OPTS)" > OPTS
	touch charm++

bgampi: bigsim AMPI

TSM: converse $(L)/libtsm.a

SM: converse $(L)/libsm.a

PVM: converse $(L)/libpvmc.a

AMR: charm++ LIBS
	cd libs/ck-libs/amr; make all OPTS='$(OPTS)'

FEM: charm++ LIBS
FEMC: FEM
FEMF: FEM
ParFUM: FEM

MBLOCK: AMPI MBLOCKC MBLOCKF

MBLOCKC: charm++ LIBS

MBLOCKF: MBLOCKC

AMPI:  charm++ LIBS
AMPIC AMPIF: AMPI

ARMCI: charm++ LIBS


f90charm: charm++ $(L)/libf90charm.a

default_libs: multicast pythonCCS

ChaNGa: charm++ cache liveViz

cache: charm++
	cd libs/ck-libs/cache && $(MAKE)

liveViz: charm++
	cd libs/ck-libs/liveViz && $(MAKE)

multicast: charmxi
	cd libs/ck-libs/multicast && $(MAKE)

pythonCCS: charmxi
	cd libs/ck-libs/pythonCCS && $(MAKE)

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
#jadeall: jade
#	make parser.g -> .java files
#	cd langs/jade && $(MAKE) p
#	make the jade.jar file
#	cd ../java/charm/jade && $(MAKE)

#include TAU defs
TAU_MAKEFILE=Make.tau
include	$(TAU_MAKEFILE) #stub-makefile

#builds TAU's trace library
Tau: 
	echo "buliding charm trace library with support for TAU."
	$(MAKE) $(L)/libtrace-Tau.a $(L)/libtrace-all.a TAU_TRACE_OBJ=libtrace-Tau.o

LIBS: CONVLIBS CHARMLIBS
	cd libs; $(MAKE) otherlibs
	echo "Built LIBS Successfully"

bgtest: bigsim AMPI
	cd ../tests ; $(MAKE) bgtest OPTS='$(OPTS)'
	cd ../examples ; $(MAKE) bgtest OPTS='$(OPTS)'

test: bigsim AMPI
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
      trace.h trace-common.h trace-bluegene.h trace-projections.h  \
      trace-simple.h trace-controlPoints.h charm-api.h \
      conv-ccs.h ccs-client.c ccs-client.h \
      ccs-server.h ccs-auth.c ccs-auth.h \
      memory-isomalloc.h debug-conv.h debug-charm.h conv-autoconfig.h \
      conv-common.h conv-config.sh conv-config.h conv-mach.h conv-mach.sh \
      blue.h blue-conv.h bgconverse.h \
	3dgridrouter.h hypercuberouter.h hypercubetopology.h	\
	routerstrategy.h convcomlibstrategy.h  convcomlib.h dummyconversestrategy.h  treerouter.h \
	convcomlibmanager.h graphrouter.h petable.h gridrouter.h router.h \
	prefixrouter.h pipebroadcastconverse.h pipelinestrategy.h cmipool.h \
      comlib.h cmidirect.h cmidirectmanytomany.h \
      controlPoints.h controlPointsf.h arrayRedistributor.h cp_effects.h register.h stats.h

# The .c files are there to be #included by clients whole
# This is a bit unusual, but makes client linking simpler.
UTILHEADERS=pup.h pupf.h pup_c.h pup_stl.h pup_mpi.h pup_toNetwork.h pup_toNetwork4.h pup_paged.h pup_cmialloc.h\
	ckimage.h ckdll.h ckhashtable.h ckbitvector.h cklists.h ckliststring.h \
	ckstatistics.h ckvector3d.h conv-lists.h RTH.h ckcomplex.h \
	sockRoutines.h sockRoutines.c ckpool.h BGLTorus.h BGPTorus.h \
	TopoManager.h XT3Torus.h XT4Torus.h cmimemcpy.h simd.h SSE-Double.h SSE-Float.h crc32.h

CKHEADERS=ck.h ckstream.h envelope.h init.h qd.h charm.h charm++.h \
	  ckfutures.h ckIgetControl.h cktiming.h \
	  ckcallback.h CkCallback.decl.h ckcallback-ccs.h 	\
	  ckarrayreductionmgr.h cknodegroupreduction.h cksection.h \
	  ckarray.h cklocation.h ckreduction.h ckcheckpoint.h ckmemcheckpoint.h ckevacuation.h\
		ckmessagelogging.h ckobjid.h\
	  ckobjQ.h readonly.h charisma.h ComlibArrayListener.h ComlibStrategy.h \
	  ComlibLearner.h $(UTILHEADERS) \
	  tempo.h waitqd.h LBDatabase.h lbdb.h lbdb++.h LBProfit.h \
          LBDBManager.h	LBComm.h LBOM.h LBObj.h LBMachineUtil.h LBAgent.h \
	  Refiner.h RefinerApprox.h RefinerComm.h ObjGraph.h ckheap.h \
          elements.h CommLBHeap.h topology.h manager.h \
	  BaseLB.h CentralLB.h CentralLBMsg.h RandCentLB.h RecBisectBfLB.h \
	  RefineLB.h RefineKLB.h RefineCommLB.h OrbLB.h \
	  GreedyLB.h GreedyCommLB.h GreedyAgentLB.h Comm1LB.h MetisLB.h \
	  TopoLB.h RefineTopoLB.h  TopoCentLB.h  PhasebyArrayLB.h RotateLB.h \
	  GridCommLB.h GridCommRefineLB.h GridMetisLB.h GridHybridLB.h GridHybridSeedLB.h \
	  DummyLB.h BlockLB.h NborBaseLB.h HybridBaseLB.h HybridLB.h HybridLBMsg.h HbmLB.h \
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
          Comm1LB.decl.h GreedyAgentLB.decl.h \
          GridCommLB.decl.h GridCommRefineLB.decl.h GridMetisLB.decl.h GridHybridLB.decl.h \
	  GridHybridSeedLB.decl.h PhasebyArrayLB.decl.h TopoLB.decl.h RefineTopoLB.decl.h \
	  TopoCentLB.decl.h NborBaseLB.decl.h NeighborLB.decl.h \
          NeighborCommLB.decl.h HbmLB.decl.h \
          HybridBaseLB.decl.h HybridLB.decl.h WSLB.decl.h EveryLB.decl.h \
          charisma.decl.h TraceSummary.decl.h TraceProjections.decl.h \
          TraceSimple.decl.h TraceControlPoints.decl.h TraceTau.decl.h \
	  TraceUtilization.decl.h BlueGene.decl.h \
	  comlib.decl.h ComlibManager.h DummyStrategy.h	\
	  MulticastStrategy.h DirectMulticastStrategy.h RingMulticastStrategy.h	\
	  ChunkMulticastStrategy.h OneTimeMulticastStrategy.h MultiRingMulticast.h \
	  EachToManyMulticastStrategy.h MsgPacker.h StreamingStrategy.h \
	  NodeMulticast.h PipeBroadcastStrategy.h MeshStreamingStrategy.h \
          PrioStreaming.h MPIStrategy.h MsgPacker.h StreamingStrategy.h \
	  BroadcastStrategy.h ComlibSectionInfo.h AAPLearner.h \
	  AAMLearner.h ComlibStats.h \
	  RectMulticastStrategy.h ckdirect.h \
	  ControlPoints.decl.h PathHistory.decl.h \
	  pathHistory.h envelope-path.h \
	  $(CVHEADERS)

ALLHEADERS=$(CKHEADERS) \
    simplemsg.h sm.h pvmc.h pvm3.h sdag.h \
	graph.h bitvecset.h ckset.h LBSimulation.h  \
	$(BLUE_HEADERS)

basics: headers
	touch basics

# Headers is used only for the dependencies
headers: dirs+sources
	$(MAKE) charmxi
	$(MAKE) headerlinks

conv-autoconfig.h: configure config_opts.sh
	@. ./config_opts.sh; echo ./configure $(QUIET) $$CONFIG_OPTS;  \
	./configure $(QUIET) $$CONFIG_OPTS

headerlinks: dirs+sources Makefile $(ALLHEADERS)
	@[ "x$QUIET" = "x--quiet" ] || echo "Soft-linking headers..."
	-@for hdr in $(ALLHEADERS) ; do test ! -f "../include/$$hdr" && ./system_ln "../tmp/$$hdr" ../include ; done
	touch headerlinks

dirs+sources:
#	rm -f .gdir
#	(cd .. ; basename "`pwd`" | sed -e 's@-.*@@') 	> .gdir
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
	./gatherflat ../../src/ck-cp           .
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
	test -f ../../src/arch/`cat .gdir`/gdir_link && cat ../../src/arch/`cat .gdir`/gdir_link > .gdir.new && ./gathertree ../../src/arch/`cat .gdir.new`      . || true
	./gatherflat ../../src/arch/`cat .vdir`      .
	./gathertree ../../examples ../examples
	./gathertree ../../tests ../tests
	rm -f ../bin/dep.pl ; cp dep.pl ../bin/
	chmod +x charmc
	-./system_ln  ../tmp/charmc ../bin/
#	-ln -s ../../java/bin charmjavabin
#	-ln -s ../../java/bin ../bin/charmjavabin
#	-ln -s charmjavabin/* ../bin/
#	-ln -s ../java ..
#	rm -f .gdir
	rm -rf ../include ; mkdir ../include
	-./system_ln  ../tmp/conv-*.*h ../include
	-./system_ln  ../tmp/cc-*.*h ../include
	if [ -x ./special.sh ] ; then ./special.sh ; fi
	if [ ! -f conv-common.h ] ; then ( touch conv-common.h ) ; fi
	touch dirs+sources

###############################################################################
#
# Converse Libraries
# 
###############################################################################
CLBLIBS=$(L)/libldb-rand.o $(L)/libldb-spray.o $(L)/libldb-neighbor.o $(L)/libldb-none.o $(L)/libldb-test.o $(L)/libldb-bluegene.o
TRACELIBS=$(L)/libtrace-projections.a  $(L)/libtrace-summary.a   $(L)/libtrace-utilization.a \
	  $(L)/libtrace-simple.a $(L)/libtrace-controlPoints.a\
          $(L)/libtrace-counter.a $(L)/libtrace-bluegene.a \
	  $(L)/libtrace-projector.a $(L)/libtrace-all.a $(L)/libtrace-converse.a \
          $(L)/libtracef_f.a $(L)/libtrace-memory.a
MEMLIBS=$(L)/libmemory-default.o $(L)/libmemory-os.o $(L)/libmemory-gnu.o \
          $(L)/libmemory-gnuold.o $(L)/libmemory-verbose.o 	\
	  $(L)/libmemory-paranoid.o \
	  $(L)/libmemory-leak.o $(L)/libmemory-isomalloc.o \
	  $(L)/libmemory-charmdebug.o $(L)/libmemory-charmdebug-mmap.o \
	  $(L)/libmemory-charmdebug-slot.o $(L)/libmemory-charmdebug-mmap-slot.o \
	  $(L)/libmemory-os-verbose.o $(L)/libmemory-os-isomalloc.o \
	  $(L)/libmemory-os-leak.o $(L)/libmemory-os-paranoid.o \
          $(L)/libmemory-os-lock.o \
	  $(L)/libmemory-os-charmdebug.o $(L)/libmemory-os-wrapper.o \
	  $(L)/libmemory-hooks-charmdebug.o
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
	debug-conv.o generate.o edgelist.o conv-ccs.o ccs-builtins.o middle-ccs.o \
	traceCore.o traceCoreCommon.o tracec.o \
	converseProjections.o machineProjections.o \
	quiescence.o isomalloc.o mem-arena.o conv-counter.o \
	global-nop.o cmipool.o cpuaffinity.o cputopology.o  \
	memoryaffinity.o modifyScheduler.o

#############################################
#Comlib Core objects that go into libck.a
COMLIB_CORE_OBJS = ComlibModuleInterface.o 
##############################################

LIBCONV_UTIL=pup_util.o pup_toNetwork.o pup_toNetwork4.o		\
	pup_xlater.o pup_c.o pup_paged.o pup_cmialloc.o			\
	ckimage.o ckdll.o ckhashtable.o sockRoutines.o			\
	conv-lists.o RTH.o persist-comm.o \
	ckbitvector.o TopoManager.o CrayNid.o crc32.o

LIBCONV_UTILF=pup_f.o

converse: fmain-ok charmrun-target swapglobal-target conv-cpm

charmrun-target: sockRoutines.c sockRoutines.h ccs-server.c ccs-server.h sockRoutines-seq.o $(L)/libck.a $(CVLIBS)
	if [ -d charmrun ] ; then ( cd charmrun ; $(MAKE) OPTS='$(OPTS)' -j1 ) ; fi
	if [ -f charmrun ] ; then ( cp -f charmrun ../bin ) ; fi

fmain-ok: fmain.f90
	-@$(CHARMC) -c fmain.f90
	-@cp fmain.o ../lib
	-@touch fmain-ok

QUICK_THREADS: $(CVHEADERS)
	-( cd QuickThreads && $(MAKE) clean )
	( cd QuickThreads;./configure )
	( cd QuickThreads ; $(MAKE) qt  OPTS='$(OPTS)' ) || ( echo "Did you try to build net-linux on a 64-bit system? Try net-linux-x86_64" && false )
	cp -f QuickThreads/libqt.a ../lib/libckqt.a
	-(CHARMINC=.; . ./conv-config.sh; \
	  test -d ../lib_so && \
	  cp -f QuickThreads/libqt.$$CMK_SHARED_SUF ../lib_so && \
	  cp -f QuickThreads/libqt.$$CMK_SHARED_SUF ../lib_so/libckqt.$$CMK_SHARED_SUF)
	touch QUICK_THREADS

METIS:  charm++
	cd libs/ck-libs/parmetis/METISLib && $(MAKE) OPTS='$(OPTS)'

CONVLIBS: converse
	cd libs; $(MAKE) convlibs OPTS='$(OPTS)'

$(L)/libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) $(TAU_LIBS) -o $@ $(LIBCONV_CORE) 

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
	$(CHARMC) -I. -o pup_toNetwork.o pup_toNetwork.C

pup_toNetwork4.o: pup_toNetwork4.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -I. -o pup_toNetwork4.o pup_toNetwork4.C

pup_xlater.o: pup_xlater.C $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -I. -o pup_xlater.o pup_xlater.C

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

TopoManager.o: TopoManager.C TopoManager.h BGLTorus.h BGPTorus.h XT3Torus.h XT4Torus.h CrayNid.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -o TopoManager.o TopoManager.C

CrayNid.o: CrayNid.c $(CVHEADERS) $(UTILHEADERS)
	$(CHARMC) -c -o CrayNid.o CrayNid.c

ckbitvector.o: ckbitvector.C ckbitvector.h $(CVHEADERS)
	$(CHARMC) -I. -o ckbitvector.o ckbitvector.C

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
	
crc32.o: crc32.c crc32.h
	$(CHARMC) -optimize -o crc32.o crc32.c

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
	$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_DEFAULT memory.c

$(L)/libmemory-os.o: $(MEM_DEPS)
	$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_OS memory.c

$(L)/libmemory-os-verbose.o: memory-verbose.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_VERBOSE memory.c || touch $@

$(L)/libmemory-os-paranoid.o: memory-paranoid.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_PARANOID memory.c || touch $@

$(L)/libmemory-os-leak.o: $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_LEAK memory.c || touch $@

$(L)/libmemory-os-isomalloc.o: memory-isomalloc.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_ISOMALLOC memory.c || touch $@

$(L)/libmemory-os-lock.o: memory-lock.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_LOCK memory.c || touch $@

$(L)/libmemory-os-charmdebug.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_OS_WRAPPED -DCMK_MEMORY_BUILD_CHARMDEBUG memory.c || touch $@

$(L)/libmemory-os-wrapper.o: memory-os-wrapper.C $(MEM_DEPS)
	-$(CHARMC) -o $@ memory-os-wrapper.C || touch $@

$(L)/libmemory-hooks-charmdebug.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -o $@ -DCMK_MEMORY_BUILD_GNU_HOOKS -DCMK_MEMORY_BUILD_CHARMDEBUG memory.c || touch $@

# If the system doesn't have sbrk, these compilations may fail.
#  This is OK, but then we can't use "-memory gnu" or friends.
$(L)/libmemory-gnu.o: $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_GNU memory.c || touch $@

$(L)/libmemory-gnuold.o: $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_GNUOLD memory.c || touch $@

$(L)/libmemory-verbose.o: memory-verbose.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_VERBOSE memory.c || touch $@

$(L)/libmemory-paranoid.o: memory-paranoid.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_PARANOID memory.c || touch $@

$(L)/libmemory-leak.o: memory-leak.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_LEAK memory.c || touch $@

$(L)/libmemory-cache.o: memory-cache.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_CACHE memory.c || touch $@

$(L)/libmemory-isomalloc.o: memory-isomalloc.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_ISOMALLOC memory.c || touch $@

$(L)/libmemory-charmdebug.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_CHARMDEBUG memory.c || touch $@

$(L)/libmemory-charmdebug-slot.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_CHARMDEBUG -DCMK_SEPARATE_SLOT memory.c || touch $@

$(L)/libmemory-charmdebug-mmap.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_CHARMDEBUG -DCPD_USE_MMAP memory.c || touch $@

$(L)/libmemory-charmdebug-mmap-slot.o: memory-charmdebug.c $(MEM_DEPS)
	-$(CHARMC) -I. -o $@ -DCMK_MEMORY_BUILD_CHARMDEBUG -DCPD_USE_MMAP -DCMK_SEPARATE_SLOT memory.c || touch $@

## Thread libraries (-thread)

LIBTHREADSDEPS = threads.c $(CVHEADERS) QUICK_THREADS

$(L)/libthreads-default.o: $(LIBTHREADSDEPS)
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_DEFAULT=1 -IQuickThreads -I. threads.c

$(L)/libthreads-qt.o: $(LIBTHREADSDEPS) 
	$(CHARMC) -o $@ -DCMK_THREADS_BUILD_QT=1 -IQuickThreads threads.c

$(L)/libthreads-context.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_CONTEXT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-uJcontext.o: $(LIBTHREADSDEPS) uJcontext.c
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_JCONTEXT=1 -IQuickThreads -I. threads.c || touch $@

$(L)/libthreads-pthreads.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_PTHREADS=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-fibers.o: $(LIBTHREADSDEPS)
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_FIBERS=1 -IQuickThreads threads.c 2> /dev/null || touch $@

$(L)/libthreads-stackcopy.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_BUILD_STACKCOPY=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-memoryalias.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_DEFAULT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-qt-memoryalias.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_QT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-context-memoryalias.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_CONTEXT=1 -IQuickThreads threads.c || touch $@

$(L)/libthreads-uJcontext-memoryalias.o: $(LIBTHREADSDEPS) 
	-$(CHARMC) -o $@ -DCMK_THREADS_ALIAS_STACK=1 -DCMK_THREADS_BUILD_JCONTEXT=1 -IQuickThreads -I. threads.c || touch $@

## Global swapping (-swapglobal)
swapglobal-target: $(L)/libglobal-swap.o $(L)/libglobal-copy.o $(L)/loadsym.o

$(L)/libglobal-swap.o: global-elfgot.C $(CVHEADERS)
	-$(CHARMC) -c global-elfgot.C -o $@ || touch $@

$(L)/libglobal-copy.o: global-elfcopy.C  $(CVHEADERS)
	-$(CHARMC) -c global-elfcopy.C -o $@ || touch $@

$(L)/loadsym.o: loadsym.c  $(CVHEADERS)
	-$(CHARMC) -c loadsym.c -o $@ || touch $@

###############################################################################
#
# Charm Libraries
#
###############################################################################

CK_LIBS_CORE=$(L)/libck.a

LIBCK_CORE=trace-common.o tracef.o init.o register.o qd.o ck.o main.o  \
	   msgalloc.o ckfutures.o ckIgetControl.o debug-message.o debug-charm.o ckcallback.o \
	   cklocation.o ckarray.o ckreduction.o ckarrayreductionmgr.o\
           tempo.o waitqd.o LBDatabase.o lbdb.o lbdbf.o charisma.o ckobjQ.o  \
	   LBAgent.o LBProfit.o ckcheckpoint.o ckmemcheckpoint.o ckevacuation.o ckmessagelogging.o\
           LBDBManager.o LBComm.o LBObj.o LBMachineUtil.o CentralPredictor.o \
	   BaseLB.o CentralLB.o HybridBaseLB.o NborBaseLB.o WSLB.o \
           ObjGraph.o graph.o LButil.o Refiner.o RefinerApprox.o  \
           RefinerComm.o bitvecset.o ckset.o ckheap.o CommLBHeap.o \
	   NullLB.o LBSimulation.o $(COMLIB_CORE_OBJS) \
	   charmProjections.o cktiming.o pathHistory.o

charm-target: converse $(L)/libck.a $(L)/libckf.a loadbalancers default_libs comlib_objs controlPoints_objs 

CHARMLIBS: charm++ CONVLIBS
	cd libs; $(MAKE) charmlibs OPTS='$(OPTS)'

$(L)/libck.a: $(LIBCK_CORE)
	$(CHARMC) -o $@ $(LIBCK_CORE)

## Charm fortran AMPI

CK_FOR_LIBS_CORE=$(L)/libckf.a

LIBCKF_CORE=charmf.o charmmod.o

$(L)/libckf.a: $(LIBCKF_CORE)
	-$(CHARMC) -o $@ $(LIBCKF_CORE) || touch $@

charmf.o:  charmf.C $(CKHEADERS) charmf.h
	$(CHARMC) -c -I. charmf.C

charmmod.o: charmmod.f90
	-$(CHARMC) -c charmmod.f90 || touch $@
	-$(CHARMC) -cpmod ../include charm.M

## Tracing libraries (profiling, -tracemode)
LIBTRACE_PROJ=trace-projections.o
$(L)/libtrace-projections.a: $(LIBTRACE_PROJ)
	$(CHARMC) -o $@ $(LIBTRACE_PROJ)

LIBTRACE_SUMM=trace-summary.o
$(L)/libtrace-summary.a: $(LIBTRACE_SUMM)
	$(CHARMC) -o $@ $(LIBTRACE_SUMM)
 
LIBTRACE_UTIL=trace-utilization.o
$(L)/libtrace-utilization.a: $(LIBTRACE_UTIL)
	$(CHARMC) -o $@ $(LIBTRACE_UTIL)

LIBTRACE_SIMPLE=trace-simple.o
$(L)/libtrace-simple.a: $(LIBTRACE_SIMPLE)
	$(CHARMC) -o $@ $(LIBTRACE_SIMPLE)

libtrace-Tau.o: trace-Tau.C charm++.h charm.h converse.h conv-config.h \
  conv-autoconfig.h conv-common.h conv-mach.h conv-mach-opt.h \
  conv-mach-ifort.h pup_c.h conv-cpm.h conv-cpath.h conv-qd.h \
  conv-random.h conv-lists.h conv-trace.h persistent.h debug-conv.h pup.h \
  middle.h middle-conv.h cklists.h ckbitvector.h ckstream.h init.h \
  ckhashtable.h debug-charm.h CkMarshall.decl.h cksection.h ckcallback.h \
  conv-ccs.h sockRoutines.h ccs-server.h ckobjQ.h ckreduction.h \
  CkReduction.decl.h cknodegroupreduction.h CkArrayReductionMgr.decl.h \
  ckmemcheckpoint.h CkMemCheckpoint.decl.h readonly.h ckarray.h \
  cklocation.h LBDatabase.h lbdb.h LBDBManager.h LBObj.h LBOM.h LBComm.h \
  LBMachineUtil.h lbdb++.h LBDatabase.decl.h NullLB.decl.h BaseLB.decl.h \
  CkLocation.decl.h CkArray.decl.h ComlibArrayListener.h ComlibStrategy.h \
  convcomlibstrategy.h ComlibLearner.h envelope.h envelope-path.h CkFutures.decl.h \
  charisma.h charisma.decl.h tempo.h tempo.decl.h waitqd.h waitqd.decl.h \
  sdag.h ckcheckpoint.h CkCheckpoint.decl.h ckevacuation.h \
  ckarrayreductionmgr.h trace.h trace-bluegene.h trace-simple.h \
  register.h trace-common.h trace-simpleBOC.h TraceSimple.decl.h \
  TraceSimple.def.h
	$(CHARMC) -c -I. $(TAU_INCLUDE) $(TAU_DEFS) trace-Tau.C -o libtrace-Tau.o

tau_selective.o: $(TAUROOT)/utils/tau_selective.cpp 
	$(CHARMC) -I$(TAUROOT)/utils -DTAU_DYNINST $(TAUROOT)/utils/tau_selective.cpp

$(L)/libtrace-Tau.a: $(TAU_TRACE_OBJ) tau_selective.o
	$(CHARMC) $(TAU_LIBS) -o $@ $(TAU_TRACE_OBJ) tau_selective.o $(TAU_TRACE_OBJ)


# Note there is a BG/P specific addition to this target in Makefile.machine
# Do not change this rule!
LIBTRACE_CONTROLPOINTS=trace-controlPoints.o
$(L)/libtrace-controlPoints.a: $(LIBTRACE_CONTROLPOINTS)
	$(CHARMC) -o $@ $^
# Do not change this rule!

LIBTRACE_COUNT=trace-counter.o
$(L)/libtrace-counter.a: $(LIBTRACE_COUNT)
	$(CHARMC) -o $@ $(LIBTRACE_COUNT)

LIBTRACE_BG=trace-bluegene.o
$(L)/libtrace-bluegene.a:$(LIBTRACE_BG)
	$(CHARMC) -o $@ $(LIBTRACE_BG)

LIBTRACE_MEMORY=trace-memory.o
$(L)/libtrace-memory.a: $(LIBTRACE_MEMORY)
	$(CHARMC) -o $@ $(LIBTRACE_MEMORY)

LIBTRACE_ALL=trace-all.o trace-projections.o trace-summary.o trace-simple.o \
$(TAU_TRACE_OBJ) trace-projector.o traceCore.o traceCoreCommon.o charmProjections.o converseProjections.o machineProjections.o trace-memory.o trace-controlPoints.o trace-utilization.o

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
TRACE_OBJS =  trace-projections.o  trace-summary.o  trace-simple.o \
	      trace-counter.o trace-controlPoints.o trace-utilization.o	\
	      trace-bluegene.o trace-projector.o trace-converse.o trace-all.o \
          trace-memory.o

######################################################################
#
#Control Point Library
#
######################################################################


CONTROLPOINTS_OBJS = controlPoints.o arrayRedistributor.o cp_effects.o

controlPoints_objs: $(L)/libmoduleControlPoints.a $(L)/libmoduleControlPointsNoTrace.a

$(L)/libmoduleControlPoints.a: $(CONTROLPOINTS_OBJS)
	$(CHARMC) -o $(L)/libmoduleControlPoints.a $(CONTROLPOINTS_OBJS)

$(L)/libmoduleControlPointsNoTrace.a: controlPointsNoTrace.o arrayRedistributor.o cp_effects.o
	$(CHARMC) -o $(L)/libmoduleControlPointsNoTrace.a controlPointsNoTrace.o arrayRedistributor.o cp_effects.o

controlPointsNoTrace.o: controlPoints.C controlPoints.h ControlPointsNoTrace.decl.h
	$(CHARMC) -I. -c controlPoints.C -o controlPointsNoTrace.o -DCP_DISABLE_TRACING $(OPTS)



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
                DummyStrategy.o NodeMulticast.o ChunkMulticastStrategy.o OneTimeMulticastStrategy.o \
		EachToManyMulticastStrategy.o ComlibSectionInfo.o \
                AAPLearner.o ComlibStats.o AAMLearner.o \
                MsgPacker.o MulticastStrategy.o DirectMulticastStrategy.o \
		RingMulticastStrategy.o MultiRingMulticast.o\
                PipeBroadcastStrategy.o MeshStreamingStrategy.o \
                PrioStreaming.o BroadcastStrategy.o RectMulticastStrategy.o \
		convcomlibstrategy.o ComlibManager.o ComlibStrategy.o 

CONVERSE_COM_OBJS = gridrouter.o hypercuberouter.o treerouter.o petable.o \
		3dgridrouter.o graphrouter.o hypercubetopology.o \
		routerstrategy.o convcomlibmanager.o pipebroadcastconverse.o \
		pipelinestrategy.o prefixrouter.o

COMLIB_OBJS = $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)

comlib_objs: $(L)/libmodulecomlib.a $(L)/libconvcom.a

$(L)/libconvcom.a: $(COMLIB_OBJS)
	$(CHARMC) -o $(L)/libconvcom.a $(CONVERSE_COM_OBJS)

#for backward compatibility, keep libmodulecommlib.a with double m.
$(L)/libmodulecomlib.a: $(COMLIB_OBJS)
	$(CHARMC) -o $(L)/libmodulecomlib.a $(COMLIB_OBJS)
	cp -f $(L)/libmodulecomlib.a $(L)/libmodulecommlib.a


###############################################################################
#
# BlueGene Emulator
#
###############################################################################

bigsim:  bigsim-target

bigemulator: bigsim

bigsim-target: charm++ $(L)/libconv-bigsim.a $(L)/libconv-bigsim-logs.a $(L)/libconv-bluegene-stub.a $(L)/libblue-standalone.a METIS

$(L)/libbluegene.a: BlueGene.C BlueGene.h BlueGene.decl.h BlueGene.def.h
	$(CHARMC) -c BlueGene.C
	$(CHARMC) -o $@ BlueGene.o

# (Converse) Blue Gene emulator
BLUE_HEADERS = blue_types.h bigsim_timing.h blue_defs.h bigsim_logs.h blue.h blue_impl.h bigsim_network.h bigsim_record.h bigsim_ooc.h bigsim_debug.h shared-alloc.h
BLUE_OBJS = blue.o bigsim_node.o bigsim_proc.o bigsim_init.o bigsim_timing.o bigsim_record.o bigsim_ooc.o shared_alloc.o
BIGSIM_IO_OBJS = bigsim_read.o bigsim_logs.o bigsim_api.o

LIBBLUE_OBJS = $(BLUE_OBJS) $(BIGSIM_IO_OBJS) blue_stub.o blue_standalone.o

$(L)/libconv-bigsim.a: $(BLUE_OBJS)
	$(CHARMC) -o $@  $(BLUE_OBJS)

$(L)/libconv-bigsim-logs.a: $(BIGSIM_IO_OBJS)
	$(CHARMC) -o $@  $(BIGSIM_IO_OBJS)

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

pvmc_buf.o: pvmc_buf.c pvmc.h pvm3.h converse.h
	$(CHARMC) -o $@ pvmc_buf.c

pvmc_conv.o: pvmc_conv.c pvmc.h pvm3.h converse.h
	$(CHARMC) -o $@ pvmc_conv.c

pvmc_pack.o: pvmc_pack.c pvmc.h pvm3.h converse.h
	$(CHARMC) -o $@ pvmc_pack.c

pvmc_comm.o: pvmc_comm.c pvmc.h pvm3.h converse.h
	$(CHARMC) -o $@ pvmc_comm.c

pvmc_groups.o: pvmc_groups.c pvmc.h pvm3.h converse.h
	$(CHARMC) -o $@ pvmc_groups.c

pvmc_main.o: pvmc_main.c pvmc.h pvm3.h converse.h
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
	@test ! -d $(DESTDIR)/include && mkdir -p $(DESTDIR)/include || true
	@test ! -d $(DESTDIR)/lib && mkdir -p $(DESTDIR)/lib || true
	@test ! -d $(DESTDIR)/lib_so && mkdir -p $(DESTDIR)/lib_so || true
	@test ! -d $(DESTDIR)/bin && mkdir -p $(DESTDIR)/bin || true
	cp -f ../include/*   $(DESTDIR)/include
	cp -f ../lib/*       $(DESTDIR)/lib
	cp -f ../lib_so/*    $(DESTDIR)/lib_so
#	cp -f ../bin/*       $(DESTDIR)/bin
	for f in ../bin/*; do if test -f $$f; then cp -f $$f $(DESTDIR)/bin; fi; done


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
	-( cd doc && $(MAKE) clean )
	( cd libs; $(MAKE) clean )
	( cd ../examples && $(MAKE) clean )
	( cd ../tests && $(MAKE) clean )

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
        for i in $(LIBCONV_CORE) $(TRACE_OBJS) $(LIBCK_CORE) $(LB_OBJ) $(LIBBLUE_OBJS) $(COMLIB_OBJS) $(CONTROLPOINTS_OBJS); do \
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
        done; \
	for cifile in $(wildcard *.ci); do \
	      echo "checking generated modules for $$cifile" ; \
	      $(CHARMXI) -M $$cifile >> $(DEPENDFILE) ; \
	done


dependsCaseSensitive: basics
	echo "Creating " $(DEPENDFILE) " ...";
	if [ -f $(DEPENDFILE) ]; then \
           /bin/cp -f $(DEPENDFILE) $(DEPENDFILE).old; \
        fi; \
	echo '#generated by make depends' > $(DEPENDFILE); \
	processFiles=''; 	\
        for i in $(LIBCONV_CORE) $(TRACE_OBJS) $(LIBCK_CORE) $(LB_OBJ) $(LIBBLUE_OBJS) $(COMLIB_OBJS) $(CONTROLPOINTS_OBJS); do \
	      processed='false';	\
	      for f in $$processFiles; do 	\
		if test $$i = $$f ; then	\
		  processed=true;  break;	\
		fi;	\
	      done;	\
	      test $$processed = 'true'  && continue;	\
	      processFiles="$$processFiles $$i";	\
	      SRCFILE=`basename $$i .o`.C ; \
	      SRCDIR=`dirname $$i` ; \
	      found=`/usr/bin/find $$SRCDIR -depth 1 -name $$SRCFILE`; \
              [ ! $$found ] && SRCFILE=`basename $$i .o`.c ; \
              echo "checking dependencies for $$SRCFILE" ; \
              if g++ -MM -Wno-deprecated -I. -I$(INC) $$SRCFILE  >> $(DEPENDFILE); then   echo '	$$(CHARMC) -c -I. '$$SRCFILE >> $(DEPENDFILE) ; else echo '' ; echo "Compilation of '$$SRCFILE' failed, please fix it first!!!!" ; exit; fi;  \
	      echo '' >> $(DEPENDFILE) ; \
        done; \
	for cifile in $(wildcard *.ci); do \
	      echo "checking generated modules for $$cifile" ; \
	      $(CHARMXI) -M $$cifile >> $(DEPENDFILE) ; \
	done



