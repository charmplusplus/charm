###############################################################################
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
###############################################################################

all:
	@echo 'try one of:'
	@echo '  make converse'
	@echo '  make charm'
	@echo '  make charm++'

charm: converse charmlibs charmxlat libcharm charmreplay

charm++: charm charmxlat++ charm++libs


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.csh)
#
###############################################################################

ALLHEADERS=acc.h c++interface.h chare.h ckdefs.h common.h communication.h condsend.h const.h core_macros.h dagger.h dtable.h env_macros.h globals.h mono.h msg_macros.h prio_macros.h quies.h stat.h sys_macros.h table.h tbl.h trans_decls.h trans_defs.h trans_externs.h user_macros.h vid.h wrtone.h converse.h internal.h quies.cplus.h performance.h dag.h generic_redn.h pglib.h

basics: $(ALLHEADERS) charmc conv-mach.csh
	if [ ! -d ../bin ] ; then mkdir ../bin ; fi
	if [ ! -d ../lib ] ; then mkdir ../lib ; fi
	rm -rf ../include
	mkdir ../include
	cp $(ALLHEADERS) ../include/
	cp charmc ../bin/
	cp conv-mach.csh ../include/
	touch basics

###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h

CVLIBS_QS=../lib/libconv-qs-fifo.o ../lib/libconv-qs-lifo.o # ../lib/libconv-qs-bfifo.o ../lib/libconv-qs-ififo.o ../lib/libconv-qs-bstack.o ../lib/libconv-qs-istack.o ../lib/libconv-qs-newfifo.o ../lib/libconv-qs-replay.o ../lib/libconv-qs-rand.o ../lib/libconv-qs-fifolifo.o

CVLIBS=../lib/libconv-core.o $(CVLIBS_QS)

converse: basics $(CVLIBS) cmk_extras

../lib/libconv-core.o: convcore.o machine.o spantree.o fifo.o
	../bin/charmc -o $@ convcore.o machine.o spantree.o fifo.o

../lib/libconv-qs-fifo.o: qs.fifo.c
	../bin/charmc -o $@ qs.fifo.c

../lib/libconv-qs-lifo.o: qs.lifo.c
	../bin/charmc -o $@ qs.lifo.c

../lib/libconv-qs-fifolifo.o: qs.fifolifo.c
	../bin/charmc -o $@ qs.fifolifo.c

../lib/libconv-qs-newfifo.o: qs.newfifo.c
	../bin/charmc -o $@ qs.newfifo.c

../lib/libconv-qs-ififo.o: qs.ififo.c
	../bin/charmc -o $@ qs.ififo.c

../lib/libconv-qs-istack.o: qs.istack.c
	../bin/charmc -o $@ qs.istack.c

../lib/libconv-qs-bfifo.o: qs.bfifo.c
	../bin/charmc -o $@ qs.bfifo.c

../lib/libconv-qs-bstack.o: qs.bstack.c
	../bin/charmc -o $@ qs.bstack.c

../lib/libconv-qs-rand.o: qs.rand.c
	../bin/charmc -o $@ qs.rand.c

../lib/libconv-qs-replay.o: qs.replay.c
	../bin/charmc -o $@ qs.replay.c

fifo.o: fifo.c $(CVHEADERS)
	../bin/charmc fifo.c

spantree.o: spantree.c $(CVHEADERS)
	../bin/charmc spantree.c

machine.o: machine.c $(CVHEADERS)
	../bin/charmc machine.c

convcore.o: convcore.c $(CVHEADERS)
	../bin/charmc convcore.c

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

CK_LDB_N=../lib/libckn-ldb-acwn.o ../lib/libckn-ldb-rand.o # ../lib/libckn-ldb-mngr.o ../lib/libckn-ldb-priomngr.o ../lib/libckn-ldb-recv.o ../lib/libckn-ldb-tok.o ../lib/libckn-ldb-btok.o

CK_LDB_D=../lib/libckd-ldb-acwn.o ../lib/libckd-ldb-rand.o # ../lib/libckd-ldb-mngr.o ../lib/libckd-ldb-priomngr.o ../lib/libckd-ldb-recv.o ../lib/libckd-ldb-tok.o ../lib/libckd-ldb-btok.o

CK_CPLUS_N=../lib/libckn-cplus-n.o

CK_CPLUS_D=../lib/libckd-cplus-n.o

CK_CORE_N=../lib/libckn-core-notrace.o ../lib/libckn-core-summary.o

CK_CORE_D=../lib/libckd-core-proj.o ../lib/libckd-core-replay.o ../lib/libckd-core-record.o

CK_LIBS_N=$(CK_LDB_N) $(CK_CPLUS_N) $(CK_CORE_N) ../lib/libckn-defmain.o

CK_LIBS_D=$(CK_LDB_D) $(CK_CPLUS_D) $(CK_CORE_D) ../lib/libckd-defmain.o

LIBCKN_CORE=common.n.o globals.n.o init.n.o register.n.o boc.n.o condsend.n.o qd.n.o vid.n.o rdonly.n.o stat.n.o wrtone.n.o acc.n.o mono.n.o tbl.n.o memstuff.n.o

LIBCKD_CORE=common.d.o globals.d.o init.d.o register.d.o boc.d.o condsend.d.o qd.d.o vid.d.o rdonly.d.o stat.d.o wrtone.d.o acc.d.o mono.d.o tbl.d.o memstuff.d.o

charmlibs: basics $(CK_LIBS_N) $(CK_LIBS_D)

LIBCKN_CORE_NOTRACE=$(LIBCKN_CORE) perf-none.n.o main.n.o ck.n.o
../lib/libckn-core-notrace.o: $(LIBCKN_CORE_NOTRACE)
	../bin/charmc -o $@ $(LIBCKN_CORE_NOTRACE)

LIBCKN_CORE_SUMMARY=$(LIBCKN_CORE) perf-sum.n.o main.n.o ck.n.o
../lib/libckn-core-summary.o: $(LIBCKN_CORE_SUMMARY)
	../bin/charmc -o $@ $(LIBCKN_CORE_SUMMARY)

LIBCKD_CORE_PROJ=$(LIBCKD_CORE) perf-proj.d.o perfio.d.o main.d.o ck.d.o
../lib/libckd-core-proj.o: $(LIBCKD_CORE_PROJ)
	../bin/charmc -o $@ $(LIBCKD_CORE_PROJ)

LIBCKD_CORE_RECORD=$(LIBCKD_CORE) perf-recd.d.o perfio.d.o main.d.o ck.d.o
../lib/libckd-core-record.o: $(LIBCKD_CORE_RECORD)
	../bin/charmc -o $@ $(LIBCKD_CORE_RECORD)

LIBCKD_CORE_REPLAY=$(LIBCKD_CORE) perf-repl.d.o perfio.d.o main.r.o ck.r.o
../lib/libckd-core-replay.o: $(LIBCKD_CORE)
	../bin/charmc -o $@ $(LIBCKD_CORE)

../lib/libckn-ldb-acwn.o ../lib/libckd-ldb-acwn.o: ldb.acwn.c
	../bin/charmc -o ../lib/libckn-ldb-acwn.o ldb.acwn.c
	../bin/charmc -o ../lib/libckd-ldb-acwn.o ldb.acwn.c -DDEBUGGING_MODE

../lib/libckn-ldb-rand.o ../lib/libckd-ldb-rand.o: ldb.rand.c
	../bin/charmc -o ../lib/libckn-ldb-rand.o ldb.rand.c
	../bin/charmc -o ../lib/libckd-ldb-rand.o ldb.rand.c -DDEBUGGING_MODE

../lib/libckn-ldb-mngr.o ../lib/libckd-ldb-mngr.o: ldb.mngr.c
	../bin/charmc -o ../lib/libckn-ldb-mngr.o ldb.mngr.c
	../bin/charmc -o ../lib/libckd-ldb-mngr.o ldb.mngr.c -DDEBUGGING_MODE

../lib/libckn-ldb-priomngr.o ../lib/libckd-ldb-priomngr.o: ldb.priomngr.c
	../bin/charmc -o ../lib/libckn-ldb-priomngr.o ldb.priomngr.c
	../bin/charmc -o ../lib/libckd-ldb-priomngr.o ldb.priomngr.c -DDEBUGGING_MODE

../lib/libckn-ldb-tok.o ../lib/libckd-ldb-tok.o: ldb.tok.c
	../bin/charmc -o ../lib/libckn-ldb-tok.o ldb.tok.c
	../bin/charmc -o ../lib/libckd-ldb-tok.o ldb.tok.c -DDEBUGGING_MODE

../lib/libckn-ldb-btok.o ../lib/libckd-ldb-btok.o: ldb.btok.c
	../bin/charmc -o ../lib/libckn-ldb-btok.o ldb.btok.c
	../bin/charmc -o ../lib/libckd-ldb-btok.o ldb.btok.c -DDEBUGGING_MODE

../lib/libckn-ldb-replay.o ../lib/libckd-ldb-replay.o: ldb.replay.c
	../bin/charmc -o ../lib/libckn-ldb-replay.o ldb.replay.c
	../bin/charmc -o ../lib/libckd-ldb-replay.o ldb.replay.c -DDEBUGGING_MODE

../lib/libckn-ldb-recv.o ../lib/libckd-ldb-recv.o: ldb.recv.c
	../bin/charmc -o ../lib/libckn-ldb-recv.o ldb.recv.c
	../bin/charmc -o ../lib/libckd-ldb-recv.o ldb.recv.c -DDEBUGGING_MODE

../lib/libckn-cplus-n.o ../lib/libckd-cplus-n.o: dummy.cplus.c
	../bin/charmc -o ../lib/libckn-cplus-n.o dummy.cplus.c
	../bin/charmc -o ../lib/libckd-cplus-n.o dummy.cplus.c -DDEBUGGING_MODE

../lib/libckn-defmain.o ../lib/libckd-defmain.o: default-main.c
	../bin/charmc -o ../lib/libckn-defmain.o default-main.c
	../bin/charmc -o ../lib/libckd-defmain.o default-main.c -DDEBUGGING_MODE

common.n.o common.d.o: common.c $(CKHEADERS)
	../bin/charmc -o common.n.o common.c
	../bin/charmc -o common.d.o common.c -DDEBUGGING_MODE

globals.n.o globals.d.o: globals.c $(CKHEADERS)
	../bin/charmc -o globals.n.o globals.c
	../bin/charmc -o globals.d.o globals.c -DDEBUGGING_MODE

register.n.o register.d.o: register.c $(CKHEADERS)
	../bin/charmc -o register.n.o register.c
	../bin/charmc -o register.d.o register.c -DDEBUGGING_MODE

init.n.o init.d.o: init.c $(CKHEADERS)
	../bin/charmc -o init.n.o init.c
	../bin/charmc -o init.d.o init.c -DDEBUGGING_MODE

boc.n.o boc.d.o: boc.c $(CKHEADERS)
	../bin/charmc -o boc.n.o boc.c
	../bin/charmc -o boc.d.o boc.c -DDEBUGGING_MODE

condsend.n.o condsend.d.o: condsend.c condsend.h $(CKHEADERS)
	../bin/charmc -o condsend.n.o condsend.c
	../bin/charmc -o condsend.d.o condsend.c -DDEBUGGING_MODE

qd.n.o qd.d.o: qd.c qd.h $(CKHEADERS)
	../bin/charmc -o qd.n.o qd.c
	../bin/charmc -o qd.d.o qd.c -DDEBUGGING_MODE

vid.n.o vid.d.o: vid.c vid.h $(CKHEADERS)
	../bin/charmc -o vid.n.o vid.c
	../bin/charmc -o vid.d.o vid.c -DDEBUGGING_MODE

stat.n.o stat.d.o: stat.c stat.h $(CKHEADERS)
	../bin/charmc -o stat.n.o stat.c
	../bin/charmc -o stat.d.o stat.c -DDEBUGGING_MODE

rdonly.n.o rdonly.d.o: rdonly.c $(CKHEADERS)
	../bin/charmc -o rdonly.n.o rdonly.c
	../bin/charmc -o rdonly.d.o rdonly.c -DDEBUGGING_MODE

acc.n.o acc.d.o: acc.c acc.h $(CKHEADERS)
	../bin/charmc -o acc.n.o acc.c
	../bin/charmc -o acc.d.o acc.c -DDEBUGGING_MODE

mono.n.o mono.d.o: mono.c mono.h $(CKHEADERS)
	../bin/charmc -o mono.n.o mono.c
	../bin/charmc -o mono.d.o mono.c -DDEBUGGING_MODE

tbl.n.o tbl.d.o: tbl.c tbl.h $(CKHEADERS)
	../bin/charmc -o tbl.n.o tbl.c
	../bin/charmc -o tbl.d.o tbl.c -DDEBUGGING_MODE

wrtone.n.o wrtone.d.o: wrtone.c wrtone.h $(CKHEADERS)
	../bin/charmc -o wrtone.n.o wrtone.c
	../bin/charmc -o wrtone.d.o wrtone.c -DDEBUGGING_MODE

memstuff.n.o memstuff.d.o: memstuff.c $(CKHEADERS)
	../bin/charmc -o memstuff.n.o memstuff.c
	../bin/charmc -o memstuff.d.o memstuff.c -DDEBUGGING_MODE

main.n.o main.d.o main.r.o: main.c $(CKHEADERS)
	../bin/charmc -o main.n.o main.c
	../bin/charmc -o main.d.o main.c -DDEBUGGING_MODE
	../bin/charmc -o main.r.o main.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

ck.n.o ck.d.o ck.r.o: ck.c $(CKHEADERS)
	../bin/charmc -o ck.n.o ck.c
	../bin/charmc -o ck.d.o ck.c -DDEBUGGING_MODE
	../bin/charmc -o ck.r.o ck.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

perfio.n.o perfio.d.o : perfio.c performance.h $(CKHEADERS)
	../bin/charmc -o perfio.n.o perfio.c
	../bin/charmc -o perfio.d.o perfio.c -DDEBUGGING_MODE

perf-none.n.o : perf-none.c performance.h $(CKHEADERS)
	../bin/charmc -o perf-none.n.o perf-none.c

perf-sum.n.o : perf-sum.c performance.h $(CKHEADERS)
	../bin/charmc -o perf-sum.n.o perf-sum.c

perf-recd.d.o : perf-recd.c performance.h $(CKHEADERS)
	../bin/charmc -o perf-recd.d.o perf-recd.c -DDEBUGGING_MODE

perf-repl.d.o : perf-repl.c performance.h $(CKHEADERS)
	../bin/charmc -o perf-repl.d.o perf-repl.c -DDEBUGGING_MODE

perf-proj.d.o : perf-proj.c performance.h $(CKHEADERS)
	../bin/charmc -o perf-proj.d.o perf-proj.c -DDEBUGGING_MODE

###############################################################################
#
# Charm++ Libraries
#
###############################################################################


charm++libs: basics ../lib/libckn-cplus-y.o ../lib/libckd-cplus-y.o 

LIBCKN_CPLUS_Y=init.cplus.n.o main.cplus.n.o ck.cplus.n.o acc.cplus.n.o mono.cplus.n.o
../lib/libckn-cplus-y.o: $(LIBCKN_CPLUS_Y)
	../bin/charmc -o $@ $(LIBCKN_CPLUS_Y)

LIBCKD_CPLUS_Y=init.cplus.d.o main.cplus.d.o ck.cplus.d.o acc.cplus.d.o mono.cplus.d.o
../lib/libckd-cplus-y.o: $(LIBCKD_CPLUS_Y)
	../bin/charmc -o $@ $(LIBCKD_CPLUS_Y)

init.cplus.n.o init.cplus.d.o : init.cplus.cxx $(CKHEADERS) c++interface.h
	../bin/charmc -o init.cplus.n.o init.cplus.cxx
	../bin/charmc -o init.cplus.d.o init.cplus.cxx -DDEBUGGING_MODE

main.cplus.n.o main.cplus.d.o : main.cplus.cxx $(CKHEADERS) c++interface.h
	../bin/charmc -o main.cplus.n.o main.cplus.cxx
	../bin/charmc -o main.cplus.d.o main.cplus.cxx -DDEBUGGING_MODE

ck.cplus.n.o ck.cplus.d.o : ck.cplus.cxx $(CKHEADERS) c++interface.h
	../bin/charmc -o ck.cplus.n.o ck.cplus.cxx
	../bin/charmc -o ck.cplus.d.o ck.cplus.cxx -DDEBUGGING_MODE

acc.cplus.n.o acc.cplus.d.o : acc.cplus.cxx $(CKHEADERS) c++interface.h acc.h
	../bin/charmc -o acc.cplus.n.o acc.cplus.cxx
	../bin/charmc -o acc.cplus.d.o acc.cplus.cxx -DDEBUGGING_MODE

mono.cplus.n.o mono.cplus.d.o : mono.cplus.cxx $(CKHEADERS) c++interface.h mono.h
	../bin/charmc -o mono.cplus.n.o mono.cplus.cxx
	../bin/charmc -o mono.cplus.d.o mono.cplus.cxx -DDEBUGGING_MODE

###############################################################################
#
# charmreplay
#
###############################################################################

charmreplay: basics ../bin/charmreplay

../bin/charmreplay: charmreplay.o perfio.n.o
	../bin/charmc -seq -o ../bin/charmreplay charmreplay.o perfio.n.o

charmreplay.o: charmreplay.c
	../bin/charmc -seq charmreplay.c

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

libcharm: basics ../lib/libcharm.a

../lib/libcharm.a: charmxlat $(LIBCHARM)
	../bin/charmc -o $@ $(LIBCHARM)

pglib.o: pglib.p pglib.h pg.int $(CKHEADERS)
	../bin/charmc pglib.p

cache.o: cache.p cache.int $(CKHEADERS)
	../bin/charmc cache.p

dag.o: dag.c dag.h $(CKHEADERS)
	../bin/charmc dag.c

barrier_redn.o: barrier_redn.p barrier_redn.int $(CKHEADERS)
	../bin/charmc barrier_redn.p

iacc_count.o: iacc_count.int $(GENERIC_ACC_H)
	../bin/charmc -o iacc_count.o GENERIC-ACC.p -DHEAD='"iacc_count.int"'

facc_count.o: facc_count.int $(GENERIC_ACC_H)
	../bin/charmc -o facc_count.o GENERIC-ACC.p -DHEAD='"facc_count.int"'

dacc_count.o: dacc_count.int $(GENERIC_ACC_H)
	../bin/charmc -o dacc_count.o GENERIC-ACC.p -DHEAD='"dacc_count.int"'

iacc_hist.o: iacc_hist.int $(GENERIC_HIST_H)
	../bin/charmc -o iacc_hist.o GENERIC-HIST.p -DHEAD='"iacc_hist.int"'

facc_hist.o: facc_hist.int $(GENERIC_HIST_H)
	../bin/charmc -o facc_hist.o GENERIC-HIST.p -DHEAD='"facc_hist.int"'

dacc_hist.o: dacc_hist.int $(GENERIC_HIST_H)
	../bin/charmc -o dacc_hist.o GENERIC-HIST.p -DHEAD='"dacc_hist.int"'

imaxredn.o: imaxredn.int $(GENERIC_REDN_H)
	../bin/charmc -o imaxredn.o GENERIC-REDN.p -DHEAD='"imaxredn.int"'

iminredn.o: iminredn.int $(GENERIC_REDN_H)
	../bin/charmc -o iminredn.o GENERIC-REDN.p -DHEAD='"iminredn.int"'

isumredn.o: isumredn.int $(GENERIC_REDN_H)
	../bin/charmc -o isumredn.o GENERIC-REDN.p -DHEAD='"isumredn.int"'

iprodredn.o: iprodredn.int $(GENERIC_REDN_H)
	../bin/charmc -o iprodredn.o GENERIC-REDN.p -DHEAD='"iprodredn.int"'

icountredn.o: icountredn.int $(GENERIC_REDN_H)
	../bin/charmc -o icountredn.o GENERIC-REDN.p -DHEAD='"icountredn.int"'

fmaxredn.o: fmaxredn.int $(GENERIC_REDN_H)
	../bin/charmc -o fmaxredn.o GENERIC-REDN.p -DHEAD='"fmaxredn.int"'

fminredn.o: fminredn.int $(GENERIC_REDN_H)
	../bin/charmc -o fminredn.o GENERIC-REDN.p -DHEAD='"fminredn.int"'

fsumredn.o: fsumredn.int $(GENERIC_REDN_H)
	../bin/charmc -o fsumredn.o GENERIC-REDN.p -DHEAD='"fsumredn.int"'

fprodredn.o: fprodredn.int $(GENERIC_REDN_H)
	../bin/charmc -o fprodredn.o GENERIC-REDN.p -DHEAD='"fprodredn.int"'

fcountredn.o: fcountredn.int $(GENERIC_REDN_H)
	../bin/charmc -o fcountredn.o GENERIC-REDN.p -DHEAD='"fcountredn.int"'

dmaxredn.o: dmaxredn.int $(GENERIC_REDN_H)
	../bin/charmc -o dmaxredn.o GENERIC-REDN.p -DHEAD='"dmaxredn.int"'

dminredn.o: dminredn.int $(GENERIC_REDN_H)
	../bin/charmc -o dminredn.o GENERIC-REDN.p -DHEAD='"dminredn.int"'

dsumredn.o: dsumredn.int $(GENERIC_REDN_H)
	../bin/charmc -o dsumredn.o GENERIC-REDN.p -DHEAD='"dsumredn.int"'

dprodredn.o: dprodredn.int $(GENERIC_REDN_H)
	../bin/charmc -o dprodredn.o GENERIC-REDN.p -DHEAD='"dprodredn.int"'

dcountredn.o: dcountredn.int $(GENERIC_REDN_H)
	../bin/charmc -o dcountredn.o GENERIC-REDN.p -DHEAD='"dcountredn.int"'

###############################################################################
#
# charmxlat and charmfilter
#
###############################################################################

CHARMXLAT=xl-main.o xl-ytab.o xl-sym.o xl-str.o xl-lex.o xl-lexyy.o xl-rdtok.o xl-parse.o xl-outh.o xl-srch.o xl-dag.o
charmxlat: basics ../bin/charmxlat

../bin/charmxlat: $(CHARMXLAT) xl-space.c
	../bin/charmc -seq -o ../bin/charmxlat $(CHARMXLAT)
	../bin/charmc -seq -o ../bin/charmfilter xl-space.c

xl-main.o: main.c
	../bin/charmc -seq xl-main.c

xl-ytab.o: xl-ytab.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-ytab.c

xl-sym.o: xl-sym.c xl-sym.h
	../bin/charmc -seq xl-sym.c

xl-str.o: xl-str.c
	../bin/charmc -seq xl-str.c

xl-lex.o: xl-lex.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-lex.c

xl-lexyy.o: xl-lexyy.c	
	../bin/charmc -seq xl-lexyy.c

xl-rdtok.o: xl-rdtok.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-rdtok.c

xl-parse.o: xl-parse.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-parse.c

xl-outh.o: xl-outh.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-outh.c

xl-srch.o: xl-srch.c xl-lex.h xl-sym.h
	../bin/charmc -seq xl-srch.c

xl-dag.o: xl-dag.c
	../bin/charmc -seq xl-dag.c

###############################################################################
#
# charmxlat++ and charmfilter++
#
###############################################################################

XPHEADERS=xp-extn.h xp-t.h xp-ytab.h

charmxlat++: basics ../bin/charmxlat++

CHARMXLATXX=xp-lexyy.o xp-process.o xp-table.o xp-ytab.o
../bin/charmxlat++: $(CHARMXLATXX) xp-space.o
	../bin/charmc -seq -o ../bin/charmxlat++ $(CHARMXLATXX)
	../bin/charmc -seq -o ../bin/charmfilter++ xp-space.o

xp-lexyy.o: xp-lexyy.c $(XPHEADERS)
	../bin/charmc -seq xp-lexyy.c

xp-process.o: xp-process.c $(XPHEADERS)
	../bin/charmc -seq xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	../bin/charmc -seq xp-table.c

xp-ytab.o: xp-ytab.c $(XPHEADERS)
	../bin/charmc -seq xp-ytab.c

xp-space.o: xp-space.c $(XPHEADERS)
	../bin/charmc -seq xp-space.c


###############################################################################
#
# Make clean
#
###############################################################################

clean: cmk_clean
	rm -f TAGS #*#
	rm -f core *.a
	rm -f core *.o
	rm -f core *~
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../include
