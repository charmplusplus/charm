##############################################################################
#
# This one Makefile serves all of converse and charm.  It includes
# rules for building:
#
#     * The CONVERSE libraries
#     * The CHARM libraries
#     * The CHARM++ libraries
#     * The CHARM translator
#     * The CHARM++ translator
#     * The conv-mach.sh and conv-mach.csh scripts
#     * The charmreplay program
# 
# This Makefile is common to all machines.  This works because the charmc
# is smart enough to "conceal" the differences between machines.  charmc
# gets its smarts by reading the conv-mach.csh configuration file.
#
# If you wish to compile with debugging, optimization, or some other charm
# option, you can do so by setting OPTS.  For example, if you need a lot
# of debugging information, you could type:
#
#     make charm OPTS="-g -save -verbose"
#
###############################################################################

all:
	@echo 'try one of:'
	@echo '  make converse'
	@echo '  make charm'
	@echo '  make charm++'

charm: converse charmxlat charmlibs libcharm charmreplay

charm++: charm charmxlat++ charm++libs

CHARMC=../bin/charmc $(OPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.csh)
#
###############################################################################

ALLHEADERS=acc.h c++interface.h chare.h ckdefs.h common.h communication.h condsend.h const.h core_macros.h dagger.h dtable.h env_macros.h globals.h mono.h msg_macros.h prio_macros.h qd.h stat.h sys_macros.h table.h tbl.h trans_decls.h trans_defs.h trans_externs.h user_macros.h vid.h wrtone.h converse.h conv-mach.h internal.h quiescence.h performance.h dag.h generic_redn.h pglib.h

ALLINTERFACES=GENERIC-ACC.int GENERIC-HIST.int GENERIC-REDN.int barrier_redn.int cache.int dacc_count.int dacc_hist.int dcountredn.int dmaxredn.int dminredn.int dprodredn.int dsumredn.int facc_count.int facc_hist.int fcountredn.int fmaxredn.int fminredn.int fprodredn.int fsumredn.int iacc_count.int iacc_hist.int icountredn.int imaxredn.int iminredn.int iprodredn.int isumredn.int pg.int

basics: dirs+sources $(ALLHEADERS) $(ALLINTERFACES) charmc conv-mach.csh
	rm -rf ../include
	mkdir ../include
	cp $(ALLHEADERS) ../include/
	cp $(ALLINTERFACES) ../include/
	rm -f ../bin/charmc ; cp charmc ../bin/
	cp conv-mach.csh ../include/
	touch basics
	
dirs+sources:
	@rm -f .vdir .gdir
	@(cd .. ; basename `pwd` > tmp/.vdir)
	@cat .vdir | sed -e 's@-.*@@' > .gdir
	if [ ! -d ../bin ] ; then mkdir ../bin ; fi
	if [ ! -d ../lib ] ; then mkdir ../lib ; fi
	rm -f ../../bin ; ln -s `cat .vdir`/bin ../../bin
	find . -type l -exec rm {} \;
	find ../../src/Common -type f \! -name "*,v" -exec ln -s {} . \;
	find ../../src/`cat .vdir` -type f \! -name "*,v" -exec ln -s {} . \;
	find ../../src/Common.`cat .gdir` -type f \! -name "*,v" -exec ln -s {} . \;
	@rm -f .vdir .gdir
	touch dirs+sources

###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-conds.h conv-mach.h 

CVLIBS_QS=libconv-qs-fifo.o libconv-qs-lifo.o # libconv-qs-bfifo.o libconv-qs-ififo.o libconv-qs-bstack.o libconv-qs-istack.o libconv-qs-newfifo.o libconv-qs-replay.o libconv-qs-rand.o libconv-qs-fifolifo.o

CVLIBS=libconv-core.o $(CVLIBS_QS)

converse: basics $(CVLIBS) cmk_extras

libconv-core.o: convcore.o conv-conds.o machine.o spantree.o fifo.o
	$(CHARMC) -cp ../lib/ -o $@ convcore.o conv-conds.o machine.o spantree.o fifo.o

libconv-qs-fifo.o: qs.fifo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.fifo.c

libconv-qs-lifo.o: qs.lifo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.lifo.c

libconv-qs-fifolifo.o: qs.fifolifo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.fifolifo.c

libconv-qs-newfifo.o: qs.newfifo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.newfifo.c

libconv-qs-ififo.o: qs.ififo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.ififo.c

libconv-qs-istack.o: qs.istack.c
	$(CHARMC) -cp ../lib/ -o $@ qs.istack.c

libconv-qs-bfifo.o: qs.bfifo.c
	$(CHARMC) -cp ../lib/ -o $@ qs.bfifo.c

libconv-qs-bstack.o: qs.bstack.c
	$(CHARMC) -cp ../lib/ -o $@ qs.bstack.c

libconv-qs-rand.o: qs.rand.c
	$(CHARMC) -cp ../lib/ -o $@ qs.rand.c

libconv-qs-replay.o: qs.replay.c
	$(CHARMC) -cp ../lib/ -o $@ qs.replay.c

fifo.o: fifo.c $(CVHEADERS)
	$(CHARMC) fifo.c

spantree.o: spantree.c $(CVHEADERS)
	$(CHARMC) spantree.c

machine.o: machine.c $(CVHEADERS)
	$(CHARMC) machine.c

convcore.o: convcore.c $(CVHEADERS)
	$(CHARMC) convcore.c

conv-conds.o: conv-conds.c $(CVHEADERS)
	$(CHARMC) conv-conds.c

converse.h:	conversedefn1.h conversetypes.h conversedefn2.h


cmk_extras:
	csh -fc 'source conv-mach.csh ; set echo ; $$CMK_EXTRAS'
	touch cmk_extras

cmk_clean:
	csh -fc 'source conv-mach.csh ; set echo ; $$CMK_CLEAN'
	rm -f cmk_extras

###############################################################################
#
# Charm Libraries
#
# Each source file is compiled twice, once with debugging enabled,
# yielding a file with the extension .d.o, and once with debugging
# disabled, yielding a file with the extension .n.o.
#
# The object files so produced are collected into libraries, which
# are named libckd-* if they contain debugging objects, or libckn-*
# if they contain normal objects.
#
###############################################################################

CKHEADERS=chare.h const.h msg_macros.h prio_macros.h env_macros.h globals.h\
	  sys_macros.h communication.h performance.h $(CVHEADERS)

CK_LDB_N=libckn-ldb-rand.o  libckn-ldbvars.o #libckn-ldb-acwn.o  libckn-ldb-mngr.o libckn-ldb-priomngr.o libckn-ldb-recv.o libckn-ldb-tok.o libckn-ldb-btok.o

CK_LDB_D= libckd-ldb-rand.o libckd-ldbvars.o #libckd-ldb-acwn.o libckd-ldb-mngr.o libckd-ldb-priomngr.o libckd-ldb-recv.o libckd-ldb-tok.o libckd-ldb-btok.o libckd-ldbvars.o

CK_CPLUS_N=libckn-cplus-n.o

CK_CPLUS_D=libckd-cplus-n.o

CK_CORE_N=libckn-core-notrace.o libckn-core-summary.o

CK_CORE_D=libckd-core-proj.o libckd-core-replay.o libckd-core-record.o

CK_LIBS_N=$(CK_LDB_N) $(CK_CPLUS_N) $(CK_CORE_N) libckn-defmain.o

CK_LIBS_D=$(CK_LDB_D) $(CK_CPLUS_D) $(CK_CORE_D) libckd-defmain.o

LIBCKN_CORE=common.n.o globals.n.o init.n.o register.n.o boc.n.o condsend.n.o qd.n.o vid.n.o rdonly.n.o stat.n.o wrtone.n.o acc.n.o mono.n.o tbl.n.o memstuff.n.o

LIBCKD_CORE=common.d.o globals.d.o init.d.o register.d.o boc.d.o condsend.d.o qd.d.o vid.d.o rdonly.d.o stat.d.o wrtone.d.o acc.d.o mono.d.o tbl.d.o memstuff.d.o

charmlibs: converse $(CK_LIBS_N) $(CK_LIBS_D)

LIBCKN_CORE_NOTRACE=$(LIBCKN_CORE) perf-none.n.o main.n.o ck.n.o
libckn-core-notrace.o: $(LIBCKN_CORE_NOTRACE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKN_CORE_NOTRACE)

LIBCKN_CORE_SUMMARY=$(LIBCKN_CORE) perf-sum.n.o main.n.o ck.n.o
libckn-core-summary.o: $(LIBCKN_CORE_SUMMARY)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKN_CORE_SUMMARY)

LIBCKD_CORE_PROJ=$(LIBCKD_CORE) perf-proj.d.o perfio.d.o main.d.o ck.d.o
libckd-core-proj.o: $(LIBCKD_CORE_PROJ)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKD_CORE_PROJ)

LIBCKD_CORE_RECORD=$(LIBCKD_CORE) perf-recd.d.o perfio.d.o main.d.o ck.d.o
libckd-core-record.o: $(LIBCKD_CORE_RECORD)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKD_CORE_RECORD)

LIBCKD_CORE_REPLAY=$(LIBCKD_CORE) perf-repl.d.o perfio.d.o main.r.o ck.r.o
libckd-core-replay.o: $(LIBCKD_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKD_CORE)

libckn-ldb-acwn.o: ldb.acwn.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.acwn.p

libckd-ldb-acwn.o: ldb.acwn.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.acwn.p -DDEBUGGING_MODE

libckn-ldb-rand.o: ldb.rand.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.rand.p

libckd-ldb-rand.o: ldb.rand.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.rand.p -DDEBUGGING_MODE

libckn-ldb-mngr.o: ldb.mngr.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.mngr.p

libckd-ldb-mngr.o: ldb.mngr.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.mngr.p -DDEBUGGING_MODE

libckn-ldb-priomngr.o: ldb.priomngr.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.priomngr.p

libckd-ldb-priomngr.o: ldb.priomngr.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.priomngr.p -DDEBUGGING_MODE

libckn-ldb-tok.o: ldb.tok.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.tok.p

libckd-ldb-tok.o: ldb.tok.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.tok.p -DDEBUGGING_MODE

libckn-ldb-btok.o: ldb.btok.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.tok.p -DBITVECTOR

libckd-ldb-btok.o: ldb.btok.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.btok.p -DDEBUGGING_MODE  -DBITVECTOR

libckn-ldb-repl.o: ldb.repl.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.repl.p

libckd-ldb-repl.o: ldb.repl.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.repl.p -DDEBUGGING_MODE

libckn-ldb-recv.o: ldb.recv.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.recv.p

libckd-ldb-recv.o: ldb.recv.p ldb.h
	$(CHARMC) -cp ../lib/ -o $@ ldb.recv.p -DDEBUGGING_MODE

ldb.h:	conversedefns.h
	
libckn-ldbvars.o: ldbvars.c
	$(CHARMC) -cp ../lib/ -o $@ ldbvars.c

libckd-ldbvars.o: ldbvars.c
	$(CHARMC) -cp ../lib/ -o $@ ldbvars.c -DDEBUGGING_MODE

libckn-cplus-n.o: dummy.cplus.c
	$(CHARMC) -cp ../lib/ -o $@ dummy.cplus.c

libckd-cplus-n.o: dummy.cplus.c
	$(CHARMC) -cp ../lib/ -o $@ dummy.cplus.c -DDEBUGGING_MODE

libckn-defmain.o: default-main.c
	$(CHARMC) -cp ../lib/ -o $@ default-main.c

libckd-defmain.o: default-main.c
	$(CHARMC) -cp ../lib/ -o $@ default-main.c -DDEBUGGING_MODE

common.n.o common.d.o: common.c $(CKHEADERS)
	$(CHARMC) -o common.n.o common.c
	$(CHARMC) -o common.d.o common.c -DDEBUGGING_MODE

globals.n.o globals.d.o: globals.c $(CKHEADERS)
	$(CHARMC) -o globals.n.o globals.c
	$(CHARMC) -o globals.d.o globals.c -DDEBUGGING_MODE

register.n.o register.d.o: register.c $(CKHEADERS)
	$(CHARMC) -o register.n.o register.c
	$(CHARMC) -o register.d.o register.c -DDEBUGGING_MODE

init.n.o init.d.o: init.c $(CKHEADERS)
	$(CHARMC) -o init.n.o init.c
	$(CHARMC) -o init.d.o init.c -DDEBUGGING_MODE

boc.n.o boc.d.o: boc.c $(CKHEADERS)
	$(CHARMC) -o boc.n.o boc.c
	$(CHARMC) -o boc.d.o boc.c -DDEBUGGING_MODE

condsend.n.o condsend.d.o: condsend.c condsend.h $(CKHEADERS)
	$(CHARMC) -o condsend.n.o condsend.c
	$(CHARMC) -o condsend.d.o condsend.c -DDEBUGGING_MODE

qd.n.o qd.d.o: qd.c qd.h $(CKHEADERS)
	$(CHARMC) -o qd.n.o qd.c
	$(CHARMC) -o qd.d.o qd.c -DDEBUGGING_MODE

vid.n.o vid.d.o: vid.c vid.h $(CKHEADERS)
	$(CHARMC) -o vid.n.o vid.c
	$(CHARMC) -o vid.d.o vid.c -DDEBUGGING_MODE

stat.n.o stat.d.o: stat.c stat.h $(CKHEADERS)
	$(CHARMC) -o stat.n.o stat.c
	$(CHARMC) -o stat.d.o stat.c -DDEBUGGING_MODE

rdonly.n.o rdonly.d.o: rdonly.c $(CKHEADERS)
	$(CHARMC) -o rdonly.n.o rdonly.c
	$(CHARMC) -o rdonly.d.o rdonly.c -DDEBUGGING_MODE

acc.n.o acc.d.o: acc.c acc.h $(CKHEADERS)
	$(CHARMC) -o acc.n.o acc.c
	$(CHARMC) -o acc.d.o acc.c -DDEBUGGING_MODE

mono.n.o mono.d.o: mono.c mono.h $(CKHEADERS)
	$(CHARMC) -o mono.n.o mono.c
	$(CHARMC) -o mono.d.o mono.c -DDEBUGGING_MODE

tbl.n.o tbl.d.o: tbl.c tbl.h $(CKHEADERS)
	$(CHARMC) -o tbl.n.o tbl.c
	$(CHARMC) -o tbl.d.o tbl.c -DDEBUGGING_MODE

wrtone.n.o wrtone.d.o: wrtone.c wrtone.h $(CKHEADERS)
	$(CHARMC) -o wrtone.n.o wrtone.c
	$(CHARMC) -o wrtone.d.o wrtone.c -DDEBUGGING_MODE

memstuff.n.o memstuff.d.o: memstuff.c $(CKHEADERS)
	$(CHARMC) -o memstuff.n.o memstuff.c
	$(CHARMC) -o memstuff.d.o memstuff.c -DDEBUGGING_MODE

main.n.o main.d.o main.r.o: main.c $(CKHEADERS)
	$(CHARMC) -o main.n.o main.c
	$(CHARMC) -o main.d.o main.c -DDEBUGGING_MODE
	$(CHARMC) -o main.r.o main.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

ck.n.o ck.d.o ck.r.o: ck.c $(CKHEADERS)
	$(CHARMC) -o ck.n.o ck.c
	$(CHARMC) -o ck.d.o ck.c -DDEBUGGING_MODE
	$(CHARMC) -o ck.r.o ck.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

perfio.n.o perfio.d.o : perfio.c performance.h $(CKHEADERS)
	$(CHARMC) -o perfio.n.o perfio.c
	$(CHARMC) -o perfio.d.o perfio.c -DDEBUGGING_MODE

perf-none.n.o : perf-none.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-none.n.o perf-none.c

perf-sum.n.o : perf-sum.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-sum.n.o perf-sum.c

perf-recd.d.o : perf-recd.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-recd.d.o perf-recd.c -DDEBUGGING_MODE

perf-repl.d.o : perf-repl.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-repl.d.o perf-repl.c -DDEBUGGING_MODE

perf-proj.d.o : perf-proj.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-proj.d.o perf-proj.c -DDEBUGGING_MODE

###############################################################################
#
# Charm++ Libraries
#
###############################################################################


charm++libs: charmlibs libckn-cplus-y.o libckd-cplus-y.o 

LIBCKN_CPLUS_Y=init.cplus.n.o main.cplus.n.o ck.cplus.n.o acc.cplus.n.o mono.cplus.n.o
libckn-cplus-y.o: $(LIBCKN_CPLUS_Y)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKN_CPLUS_Y)

LIBCKD_CPLUS_Y=init.cplus.d.o main.cplus.d.o ck.cplus.d.o acc.cplus.d.o mono.cplus.d.o
libckd-cplus-y.o: $(LIBCKD_CPLUS_Y)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCKD_CPLUS_Y)

init.cplus.n.o init.cplus.d.o : init.cplus.cxx $(CKHEADERS) c++interface.h
	$(CHARMC) -o init.cplus.n.o init.cplus.cxx
	$(CHARMC) -o init.cplus.d.o init.cplus.cxx -DDEBUGGING_MODE

main.cplus.n.o main.cplus.d.o : main.cplus.cxx $(CKHEADERS) c++interface.h
	$(CHARMC) -o main.cplus.n.o main.cplus.cxx
	$(CHARMC) -o main.cplus.d.o main.cplus.cxx -DDEBUGGING_MODE

ck.cplus.n.o ck.cplus.d.o : ck.cplus.cxx $(CKHEADERS) c++interface.h
	$(CHARMC) -o ck.cplus.n.o ck.cplus.cxx
	$(CHARMC) -o ck.cplus.d.o ck.cplus.cxx -DDEBUGGING_MODE

acc.cplus.n.o acc.cplus.d.o : acc.cplus.cxx $(CKHEADERS) c++interface.h acc.h
	$(CHARMC) -o acc.cplus.n.o acc.cplus.cxx
	$(CHARMC) -o acc.cplus.d.o acc.cplus.cxx -DDEBUGGING_MODE

mono.cplus.n.o mono.cplus.d.o : mono.cplus.cxx $(CKHEADERS) c++interface.h mono.h
	$(CHARMC) -o mono.cplus.n.o mono.cplus.cxx
	$(CHARMC) -o mono.cplus.d.o mono.cplus.cxx -DDEBUGGING_MODE

###############################################################################
#
# charmreplay
#
###############################################################################

charmreplay: basics charmreplay.c perfio.c
	$(CHARMC) -seq -o $@ -cp ../bin/ charmreplay.c perfio.c

###############################################################################
#
# libcharm
#
###############################################################################

GENERIC_ACC_H=GENERIC-ACC.int GENERIC-ACC.p $(CKHEADERS)
GENERIC_HIST_H=GENERIC-HIST.int GENERIC-HIST.p $(CKHEADERS)
GENERIC_REDN_H=GENERIC-REDN.int GENERIC-REDN.p $(CKHEADERS)

LIBCHARM=pglib.o cache.o dag.o barrier_redn.o \
	 iacc_count.o facc_count.o dacc_count.o \
         iacc_hist.o facc_hist.o dacc_hist.o \
         iminredn.o imaxredn.o isumredn.o iprodredn.o icountredn.o \
	 fminredn.o fmaxredn.o fsumredn.o fprodredn.o \
	 dminredn.o dmaxredn.o dsumredn.o dprodredn.o

libcharm: basics charmxlat libcharm.a

libcharm.a: $(LIBCHARM)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCHARM)

pglib.o: pglib.p pglib.h pg.int $(CKHEADERS)
	$(CHARMC) pglib.p

cache.o: cache.p cache.int $(CKHEADERS)
	$(CHARMC) cache.p

dag.o: dag.c dag.h $(CKHEADERS)
	$(CHARMC) dag.c

barrier_redn.o: barrier_redn.p barrier_redn.int $(CKHEADERS)
	$(CHARMC) barrier_redn.p

iacc_count.o: iacc_count.int $(GENERIC_ACC_H)
	$(CHARMC) -o iacc_count.o GENERIC-ACC.p -DHEAD='"iacc_count.int"'

facc_count.o: facc_count.int $(GENERIC_ACC_H)
	$(CHARMC) -o facc_count.o GENERIC-ACC.p -DHEAD='"facc_count.int"'

dacc_count.o: dacc_count.int $(GENERIC_ACC_H)
	$(CHARMC) -o dacc_count.o GENERIC-ACC.p -DHEAD='"dacc_count.int"'

iacc_hist.o: iacc_hist.int $(GENERIC_HIST_H)
	$(CHARMC) -o iacc_hist.o GENERIC-HIST.p -DHEAD='"iacc_hist.int"'

facc_hist.o: facc_hist.int $(GENERIC_HIST_H)
	$(CHARMC) -o facc_hist.o GENERIC-HIST.p -DHEAD='"facc_hist.int"'

dacc_hist.o: dacc_hist.int $(GENERIC_HIST_H)
	$(CHARMC) -o dacc_hist.o GENERIC-HIST.p -DHEAD='"dacc_hist.int"'

imaxredn.o: imaxredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o imaxredn.o GENERIC-REDN.p -DHEAD='"imaxredn.int"'

iminredn.o: iminredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o iminredn.o GENERIC-REDN.p -DHEAD='"iminredn.int"'

isumredn.o: isumredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o isumredn.o GENERIC-REDN.p -DHEAD='"isumredn.int"'

iprodredn.o: iprodredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o iprodredn.o GENERIC-REDN.p -DHEAD='"iprodredn.int"'

icountredn.o: icountredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o icountredn.o GENERIC-REDN.p -DHEAD='"icountredn.int"'

fmaxredn.o: fmaxredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o fmaxredn.o GENERIC-REDN.p -DHEAD='"fmaxredn.int"'

fminredn.o: fminredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o fminredn.o GENERIC-REDN.p -DHEAD='"fminredn.int"'

fsumredn.o: fsumredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o fsumredn.o GENERIC-REDN.p -DHEAD='"fsumredn.int"'

fprodredn.o: fprodredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o fprodredn.o GENERIC-REDN.p -DHEAD='"fprodredn.int"'

fcountredn.o: fcountredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o fcountredn.o GENERIC-REDN.p -DHEAD='"fcountredn.int"'

dmaxredn.o: dmaxredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o dmaxredn.o GENERIC-REDN.p -DHEAD='"dmaxredn.int"'

dminredn.o: dminredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o dminredn.o GENERIC-REDN.p -DHEAD='"dminredn.int"'

dsumredn.o: dsumredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o dsumredn.o GENERIC-REDN.p -DHEAD='"dsumredn.int"'

dprodredn.o: dprodredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o dprodredn.o GENERIC-REDN.p -DHEAD='"dprodredn.int"'

dcountredn.o: dcountredn.int $(GENERIC_REDN_H)
	$(CHARMC) -o dcountredn.o GENERIC-REDN.p -DHEAD='"dcountredn.int"'

###############################################################################
#
# charmxlat and charmfilter
#
###############################################################################

CHARMXLAT=xl-main.o xl-ytab.o xl-sym.o xl-str.o xl-lex.o xl-lexyy.o xl-rdtok.o xl-parse.o xl-outh.o xl-srch.o xl-dag.o

charmxlat charmfilter: basics $(CHARMXLAT)
	$(CHARMC) -seq -o charmxlat   -cp ../bin/ $(CHARMXLAT)
	$(CHARMC) -seq -o charmfilter -cp ../bin/ xl-space.c

xl-main.o: main.c
	$(CHARMC) -seq xl-main.c

xl-ytab.o: xl-ytab.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-ytab.c

xl-sym.o: xl-sym.c xl-sym.h
	$(CHARMC) -seq xl-sym.c

xl-str.o: xl-str.c
	$(CHARMC) -seq xl-str.c

xl-lex.o: xl-lex.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-lex.c

xl-lexyy.o: xl-lexyy.c	
	$(CHARMC) -seq xl-lexyy.c

xl-rdtok.o: xl-rdtok.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-rdtok.c

xl-parse.o: xl-parse.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-parse.c

xl-outh.o: xl-outh.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-outh.c

xl-srch.o: xl-srch.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-srch.c

xl-dag.o: xl-dag.c
	$(CHARMC) -seq xl-dag.c

###############################################################################
#
# charmxlat++ and charmfilter++
#
###############################################################################

XPHEADERS=xp-extn.h xp-t.h xp-ytab.h

CHARMXLATXX=xp-lexyy.o xp-process.o xp-table.o xp-ytab.o

charmxlat++: basics $(CHARMXLATXX) xp-space.o
	$(CHARMC) -seq -o charmxlat++   -cp ../bin/ $(CHARMXLATXX)
	$(CHARMC) -seq -o charmfilter++ -cp ../bin/ xp-space.o

xp-lexyy.o: xp-lexyy.c $(XPHEADERS)
	$(CHARMC) -seq xp-lexyy.c

xp-process.o: xp-process.c $(XPHEADERS)
	$(CHARMC) -seq xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	$(CHARMC) -seq xp-table.c

xp-ytab.o: xp-ytab.c $(XPHEADERS)
	$(CHARMC) -seq xp-ytab.c

xp-space.o: xp-space.c $(XPHEADERS)
	$(CHARMC) -seq xp-space.c


###############################################################################
#
# Make clean
#
###############################################################################

clean: cmk_clean
	rm -f TAGS #*#
	rm -f basics *.a
	rm -f basics *.o
	rm -f basics *~
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../include


