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

charm: converse charmlibs charmxlat libcharm.a charmreplay

charm++: charm charmxlat++ charm++libs

###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h

CVLIBS_QS=libconv-qs-fifo.o libconv-qs-lifo.o # libconv-qs-bfifo.o libconv-qs-ififo.o libconv-qs-bstack.o libconv-qs-istack.o libconv-qs-newfifo.o libconv-qs-replay.o libconv-qs-rand.o libconv-qs-fifolifo.o

CVLIBS=libconv-core.o $(CVLIBS_QS)

converse: $(CVLIBS) cmk_extras

libconv-core.o: convcore.o machine.o spantree.o fifo.o
	charmc -cwd -o libconv-core.o convcore.o machine.o spantree.o fifo.o

libconv-qs-fifo.o: qs.fifo.c
	charmc -cwd -o libconv-qs-fifo.o qs.fifo.c

libconv-qs-lifo.o: qs.lifo.c
	charmc -cwd -o libconv-qs-lifo.o qs.lifo.c

libconv-qs-fifolifo.o: qs.fifolifo.c
	charmc -cwd -o libconv-qs-fifolifo.o qs.fifolifo.c

libconv-qs-newfifo.o: qs.newfifo.c
	charmc -cwd -o libconv-qs-newfifo.o qs.newfifo.c

libconv-qs-ififo.o: qs.ififo.c
	charmc -cwd -o libconv-qs-ififo.o qs.ififo.c

libconv-qs-istack.o: qs.istack.c
	charmc -cwd -o libconv-qs-istack.o qs.istack.c

libconv-qs-bfifo.o: qs.bfifo.c
	charmc -cwd -o libconv-qs-bfifo.o qs.bfifo.c

libconv-qs-bstack.o: qs.bstack.c
	charmc -cwd -o libconv-qs-bstack.o qs.bstack.c

libconv-qs-rand.o: qs.rand.c
	charmc -cwd -o libconv-qs-rand.o qs.rand.c

libconv-qs-replay.o: qs.replay.c
	charmc -cwd -o libconv-qs-replay.o qs.replay.c

fifo.o: fifo.c $(CVHEADERS)
	charmc -cwd fifo.c

spantree.o: spantree.c $(CVHEADERS)
	charmc -cwd spantree.c

machine.o: machine.c $(CVHEADERS)
	charmc -cwd machine.c

convcore.o: convcore.c $(CVHEADERS)
	charmc -cwd convcore.c

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

LIBCK_CORE_N=common.n.o globals.n.o init.n.o register.n.o boc.n.o condsend.n.o qd.n.o vid.n.o rdonly.n.o stat.n.o wrtone.n.o acc.n.o mono.n.o tbl.n.o memstuff.n.o

LIBCK_CORE_D=common.d.o globals.d.o init.d.o register.d.o boc.d.o condsend.d.o qd.d.o vid.d.o rdonly.d.o stat.d.o wrtone.d.o acc.d.o mono.d.o tbl.d.o memstuff.d.o

CK_LDB_N=libckn-ldb-acwn.o libckn-ldb-rand.o # libckn-ldb-mngr.o libckn-ldb-priomngr.o libckn-ldb-recv.o libckn-ldb-tok.o libckn-ldb-btok.o

CK_LDB_D=libckd-ldb-acwn.o libckd-ldb-rand.o # libckd-ldb-mngr.o libckd-ldb-priomngr.o libckd-ldb-recv.o libckd-ldb-tok.o libckd-ldb-btok.o

CK_CPLUS_N=libckn-cplus-n.o

CK_CPLUS_D=libckd-cplus-n.o

CK_CORE_N=libckn-core-notrace.o libckn-core-summary.o

CK_CORE_D=libckd-core-proj.o libckd-core-replay.o libckd-core-record.o

CK_LIBS_N=$(CK_LDB_N) $(CK_CPLUS_N) $(CK_CORE_N) libckn-defmain.o

CK_LIBS_D=$(CK_LDB_D) $(CK_CPLUS_D) $(CK_CORE_D) libckd-defmain.o

charmlibs: $(CK_LIBS_N) $(CK_LIBS_D)

libckn-core-notrace.o: $(LIBCK_CORE_N) perf-none.n.o main.n.o ck.n.o
	charmc -cwd -o $@ $(LIBCK_CORE_N) perf-none.n.o main.n.o ck.n.o

libckn-core-summary.o:$(LIBCK_CORE_N) perf-sum.n.o main.n.o ck.n.o
	charmc -cwd -o $@ $(LIBCK_CORE_N) perf-sum.n.o main.n.o ck.n.o

libckd-core-proj.o:$(LIBCK_CORE_D) perf-proj.d.o perfio.d.o main.d.o ck.d.o
	charmc -cwd -o $@ $(LIBCK_CORE_D) perf-proj.d.o perfio.d.o main.d.o ck.d.o

libckd-core-record.o:$(LIBCK_CORE_D) perf-recd.d.o perfio.d.o main.d.o ck.d.o
	charmc -cwd -o $@ $(LIBCK_CORE_D) perf-recd.d.o perfio.d.o main.d.o ck.d.o

libckd-core-replay.o:$(CK_CORE_D_OBJS) perf-repl.d.o perfio.d.o main.r.o ck.r.o
	charmc -cwd -o $@ $(LIBCK_CORE_D) perf-repl.d.o perfio.d.o main.r.o ck.r.o

libckn-ldb-acwn.o libckd-ldb-acwn.o: ldb.acwn.c
	charmc -cwd -o libckn-ldb-acwn.o ldb.acwn.c
	charmc -cwd -o libckd-ldb-acwn.o ldb.acwn.c -DDEBUGGING_MODE

libckn-ldb-rand.o libckd-ldb-rand.o: ldb.rand.c
	charmc -cwd -o libckn-ldb-rand.o ldb.rand.c
	charmc -cwd -o libckd-ldb-rand.o ldb.rand.c -DDEBUGGING_MODE

libckn-ldb-mngr.o libckd-ldb-mngr.o: ldb.mngr.c
	charmc -cwd -o libckn-ldb-mngr.o ldb.mngr.c
	charmc -cwd -o libckd-ldb-mngr.o ldb.mngr.c -DDEBUGGING_MODE

libckn-ldb-priomngr.o libckd-ldb-priomngr.o: ldb.priomngr.c
	charmc -cwd -o libckn-ldb-priomngr.o ldb.priomngr.c
	charmc -cwd -o libckd-ldb-priomngr.o ldb.priomngr.c -DDEBUGGING_MODE

libckn-ldb-tok.o libckd-ldb-tok.o: ldb.tok.c
	charmc -cwd -o libckn-ldb-tok.o ldb.tok.c
	charmc -cwd -o libckd-ldb-tok.o ldb.tok.c -DDEBUGGING_MODE

libckn-ldb-btok.o libckd-ldb-btok.o: ldb.btok.c
	charmc -cwd -o libckn-ldb-btok.o ldb.btok.c
	charmc -cwd -o libckd-ldb-btok.o ldb.btok.c -DDEBUGGING_MODE

libckn-ldb-replay.o libckd-ldb-replay.o: ldb.replay.c
	charmc -cwd -o libckn-ldb-replay.o ldb.replay.c
	charmc -cwd -o libckd-ldb-replay.o ldb.replay.c -DDEBUGGING_MODE

libckn-ldb-recv.o libckd-ldb-recv.o: ldb.recv.c
	charmc -cwd -o libckn-ldb-recv.o ldb.recv.c
	charmc -cwd -o libckd-ldb-recv.o ldb.recv.c -DDEBUGGING_MODE

libckn-cplus-n.o libckd-cplus-n.o: dummy.cplus.c
	charmc -cwd -o libckn-cplus-n.o dummy.cplus.c
	charmc -cwd -o libckd-cplus-n.o dummy.cplus.c -DDEBUGGING_MODE

libckn-defmain.o libckd-defmain.o: default-main.c
	charmc -cwd -o libckn-defmain.o default-main.c
	charmc -cwd -o libckd-defmain.o default-main.c -DDEBUGGING_MODE

common.n.o common.d.o: common.c $(CKHEADERS)
	charmc -cwd -o common.n.o common.c
	charmc -cwd -o common.d.o common.c -DDEBUGGING_MODE

globals.n.o globals.d.o: globals.c $(CKHEADERS)
	charmc -cwd -o globals.n.o globals.c
	charmc -cwd -o globals.d.o globals.c -DDEBUGGING_MODE

register.n.o register.d.o: register.c $(CKHEADERS)
	charmc -cwd -o register.n.o register.c
	charmc -cwd -o register.d.o register.c -DDEBUGGING_MODE

init.n.o init.d.o: init.c $(CKHEADERS)
	charmc -cwd -o init.n.o init.c
	charmc -cwd -o init.d.o init.c -DDEBUGGING_MODE

boc.n.o boc.d.o: boc.c $(CKHEADERS)
	charmc -cwd -o boc.n.o boc.c
	charmc -cwd -o boc.d.o boc.c -DDEBUGGING_MODE

condsend.n.o condsend.d.o: condsend.c condsend.h $(CKHEADERS)
	charmc -cwd -o condsend.n.o condsend.c
	charmc -cwd -o condsend.d.o condsend.c -DDEBUGGING_MODE

qd.n.o qd.d.o: qd.c qd.h $(CKHEADERS)
	charmc -cwd -o qd.n.o qd.c
	charmc -cwd -o qd.d.o qd.c -DDEBUGGING_MODE

vid.n.o vid.d.o: vid.c vid.h $(CKHEADERS)
	charmc -cwd -o vid.n.o vid.c
	charmc -cwd -o vid.d.o vid.c -DDEBUGGING_MODE

stat.n.o stat.d.o: stat.c stat.h $(CKHEADERS)
	charmc -cwd -o stat.n.o stat.c
	charmc -cwd -o stat.d.o stat.c -DDEBUGGING_MODE

rdonly.n.o rdonly.d.o: rdonly.c $(CKHEADERS)
	charmc -cwd -o rdonly.n.o rdonly.c
	charmc -cwd -o rdonly.d.o rdonly.c -DDEBUGGING_MODE

acc.n.o acc.d.o: acc.c acc.h $(CKHEADERS)
	charmc -cwd -o acc.n.o acc.c
	charmc -cwd -o acc.d.o acc.c -DDEBUGGING_MODE

mono.n.o mono.d.o: mono.c mono.h $(CKHEADERS)
	charmc -cwd -o mono.n.o mono.c
	charmc -cwd -o mono.d.o mono.c -DDEBUGGING_MODE

tbl.n.o tbl.d.o: tbl.c tbl.h $(CKHEADERS)
	charmc -cwd -o tbl.n.o tbl.c
	charmc -cwd -o tbl.d.o tbl.c -DDEBUGGING_MODE

wrtone.n.o wrtone.d.o: wrtone.c wrtone.h $(CKHEADERS)
	charmc -cwd -o wrtone.n.o wrtone.c
	charmc -cwd -o wrtone.d.o wrtone.c -DDEBUGGING_MODE

memstuff.n.o memstuff.d.o: memstuff.c $(CKHEADERS)
	charmc -cwd -o memstuff.n.o memstuff.c
	charmc -cwd -o memstuff.d.o memstuff.c -DDEBUGGING_MODE

main.n.o main.d.o main.r.o: main.c $(CKHEADERS)
	charmc -cwd -o main.n.o main.c
	charmc -cwd -o main.d.o main.c -DDEBUGGING_MODE
	charmc -cwd -o main.r.o main.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

ck.n.o ck.d.o ck.r.o: ck.c $(CKHEADERS)
	charmc -cwd -o ck.n.o ck.c
	charmc -cwd -o ck.d.o ck.c -DDEBUGGING_MODE
	charmc -cwd -o ck.r.o ck.c -DDEBUGGING_MODE -DREPLAY_DEBUGGING

perfio.n.o perfio.d.o : perfio.c performance.h $(CKHEADERS)
	charmc -cwd -o perfio.n.o perfio.c
	charmc -cwd -o perfio.d.o perfio.c -DDEBUGGING_MODE

perf-none.n.o : perf-none.c performance.h $(CKHEADERS)
	charmc -cwd -o perf-none.n.o perf-none.c

perf-sum.n.o : perf-sum.c performance.h $(CKHEADERS)
	charmc -cwd -o perf-sum.n.o perf-sum.c

perf-recd.d.o : perf-recd.c performance.h $(CKHEADERS)
	charmc -cwd -o perf-recd.d.o perf-recd.c -DDEBUGGING_MODE

perf-repl.d.o : perf-repl.c performance.h $(CKHEADERS)
	charmc -cwd -o perf-repl.d.o perf-repl.c -DDEBUGGING_MODE

perf-proj.d.o : perf-proj.c performance.h $(CKHEADERS)
	charmc -cwd -o perf-proj.d.o perf-proj.c -DDEBUGGING_MODE

###############################################################################
#
# Charm++ Libraries
#
###############################################################################


LIBCP_N=init.cplus.n.o main.cplus.n.o ck.cplus.n.o acc.cplus.n.o mono.cplus.n.o
LIBCP_D=init.cplus.d.o main.cplus.d.o ck.cplus.d.o acc.cplus.d.o mono.cplus.d.o

charm++libs: libckn-cplus-y.o libckd-cplus-n.o 

libckn-cplus-y.o: $(LIBCP_N)
	charmc -cwd -o $@ $(LIBCP_N)

libckd-cplus-y.o: $(LIBCP_D)
	charmc -cwd -o $@ $(LIBCP_D)

init.cplus.n.o init.cplus.d.o : init.cplus.cxx $(CKHEADERS) c++interface.h
	charmc -cwd -o init.cplus.n.o init.cplus.cxx
	charmc -cwd -o init.cplus.d.o init.cplus.cxx -DDEBUGGING_MODE

main.cplus.n.o main.cplus.d.o : main.cplus.cxx $(CKHEADERS) c++interface.h
	charmc -cwd -o main.cplus.n.o main.cplus.cxx
	charmc -cwd -o main.cplus.d.o main.cplus.cxx -DDEBUGGING_MODE

ck.cplus.n.o ck.cplus.d.o : ck.cplus.cxx $(CKHEADERS) c++interface.h
	charmc -cwd -o ck.cplus.n.o ck.cplus.cxx
	charmc -cwd -o ck.cplus.d.o ck.cplus.cxx -DDEBUGGING_MODE

acc.cplus.n.o acc.cplus.d.o : acc.cplus.cxx $(CKHEADERS) c++interface.h acc.h
	charmc -cwd -o acc.cplus.n.o acc.cplus.cxx
	charmc -cwd -o acc.cplus.d.o acc.cplus.cxx -DDEBUGGING_MODE

mono.cplus.n.o mono.cplus.d.o : mono.cplus.cxx $(CKHEADERS) c++interface.h mono.h
	charmc -cwd -o mono.cplus.n.o mono.cplus.cxx
	charmc -cwd -o mono.cplus.d.o mono.cplus.cxx -DDEBUGGING_MODE

###############################################################################
#
# charmreplay
#
###############################################################################

charmreplay: charmreplay.o perfio.n.o
	charmc -seq -cwd -o charmreplay charmreplay.o perfio.n.o

charmreplay.o: charmreplay.c
	charmc -seq -cwd charmreplay.c

###############################################################################
#
# libcharm
#
###############################################################################

LIBCHARM=pglib.o cache.o dag.o barrier_redn.o \
	 iacc_count.o facc_count.o dacc_count.o \
         iacc_hist.o facc_hist.o dacc_hist.o \
         iminredn.o imaxredn.o isumredn.o iprodredn.o icountredn.o \
	 fminredn.o fmaxredn.o fsumredn.o fprodredn.o \
	 dminredn.o dmaxredn.o dsumredn.o dprodredn.o

GENERIC_ACC_H=GENERIC-ACC.int GENERIC-ACC.p $(CKHEADERS)
GENERIC_HIST_H=GENERIC-HIST.int GENERIC-HIST.p $(CKHEADERS)
GENERIC_REDN_H=GENERIC-REDN.int GENERIC-REDN.p $(CKHEADERS)

libcharm.a: charmxlat $(LIBCHARM)
	charmc -cwd -o libcharm.a $(LIBCHARM)

pglib.o: pglib.p pglib.h pg.int $(CKHEADERS)
	charmc -cwd pglib.p

cache.o: cache.p cache.int $(CKHEADERS)
	charmc -cwd cache.p

dag.o: dag.c dag.h $(CKHEADERS)
	charmc -cwd dag.c

barrier_redn.o: barrier_redn.p barrier_redn.int $(CKHEADERS)
	charmc -cwd barrier_redn.p

iacc_count.o: iacc_count.int $(GENERIC_ACC_H)
	charmc -cwd -o iacc_count.o GENERIC-ACC.p -DHEAD='"iacc_count.int"'

facc_count.o: facc_count.int $(GENERIC_ACC_H)
	charmc -cwd -o facc_count.o GENERIC-ACC.p -DHEAD='"facc_count.int"'

dacc_count.o: dacc_count.int $(GENERIC_ACC_H)
	charmc -cwd -o dacc_count.o GENERIC-ACC.p -DHEAD='"dacc_count.int"'

iacc_hist.o: iacc_hist.int $(GENERIC_HIST_H)
	charmc -cwd -o iacc_hist.o GENERIC-HIST.p -DHEAD='"iacc_hist.int"'

facc_hist.o: facc_hist.int $(GENERIC_HIST_H)
	charmc -cwd -o facc_hist.o GENERIC-HIST.p -DHEAD='"facc_hist.int"'

dacc_hist.o: dacc_hist.int $(GENERIC_HIST_H)
	charmc -cwd -o dacc_hist.o GENERIC-HIST.p -DHEAD='"dacc_hist.int"'

imaxredn.o: imaxredn.int $(GENERIC_REDN_H)
	charmc -cwd -o imaxredn.o GENERIC-REDN.p -DHEAD='"imaxredn.int"'

iminredn.o: iminredn.int $(GENERIC_REDN_H)
	charmc -cwd -o iminredn.o GENERIC-REDN.p -DHEAD='"iminredn.int"'

isumredn.o: isumredn.int $(GENERIC_REDN_H)
	charmc -cwd -o isumredn.o GENERIC-REDN.p -DHEAD='"isumredn.int"'

iprodredn.o: iprodredn.int $(GENERIC_REDN_H)
	charmc -cwd -o iprodredn.o GENERIC-REDN.p -DHEAD='"iprodredn.int"'

icountredn.o: icountredn.int $(GENERIC_REDN_H)
	charmc -cwd -o icountredn.o GENERIC-REDN.p -DHEAD='"icountredn.int"'

fmaxredn.o: fmaxredn.int $(GENERIC_REDN_H)
	charmc -cwd -o fmaxredn.o GENERIC-REDN.p -DHEAD='"fmaxredn.int"'

fminredn.o: fminredn.int $(GENERIC_REDN_H)
	charmc -cwd -o fminredn.o GENERIC-REDN.p -DHEAD='"fminredn.int"'

fsumredn.o: fsumredn.int $(GENERIC_REDN_H)
	charmc -cwd -o fsumredn.o GENERIC-REDN.p -DHEAD='"fsumredn.int"'

fprodredn.o: fprodredn.int $(GENERIC_REDN_H)
	charmc -cwd -o fprodredn.o GENERIC-REDN.p -DHEAD='"fprodredn.int"'

fcountredn.o: fcountredn.int $(GENERIC_REDN_H)
	charmc -cwd -o fcountredn.o GENERIC-REDN.p -DHEAD='"fcountredn.int"'

dmaxredn.o: dmaxredn.int $(GENERIC_REDN_H)
	charmc -cwd -o dmaxredn.o GENERIC-REDN.p -DHEAD='"dmaxredn.int"'

dminredn.o: dminredn.int $(GENERIC_REDN_H)
	charmc -cwd -o dminredn.o GENERIC-REDN.p -DHEAD='"dminredn.int"'

dsumredn.o: dsumredn.int $(GENERIC_REDN_H)
	charmc -cwd -o dsumredn.o GENERIC-REDN.p -DHEAD='"dsumredn.int"'

dprodredn.o: dprodredn.int $(GENERIC_REDN_H)
	charmc -cwd -o dprodredn.o GENERIC-REDN.p -DHEAD='"dprodredn.int"'

dcountredn.o: dcountredn.int $(GENERIC_REDN_H)
	charmc -cwd -o dcountredn.o GENERIC-REDN.p -DHEAD='"dcountredn.int"'

###############################################################################
#
# charmxlat and charmfilter
#
###############################################################################

CHARMXLAT=xl-main.o xl-ytab.o xl-sym.o xl-str.o xl-lex.o xl-lexyy.o xl-rdtok.o xl-parse.o xl-outh.o xl-srch.o xl-dag.o
charmxlat: $(CHARMXLAT) xl-space.c
	charmc -seq -cwd -o charmxlat $(CHARMXLAT)
	charmc -seq -cwd -o charmfilter xl-space.c

xl-main.o: main.c
	charmc -seq -cwd xl-main.c

xl-ytab.o: xl-ytab.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-ytab.c

xl-sym.o: xl-sym.c xl-sym.h
	charmc -seq -cwd xl-sym.c

xl-str.o: xl-str.c
	charmc -seq -cwd xl-str.c

xl-lex.o: xl-lex.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-lex.c

xl-lexyy.o: xl-lexyy.c	
	charmc -seq -cwd xl-lexyy.c

xl-rdtok.o: xl-rdtok.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-rdtok.c

xl-parse.o: xl-parse.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-parse.c

xl-outh.o: xl-outh.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-outh.c

xl-srch.o: xl-srch.c xl-lex.h xl-sym.h
	charmc -seq -cwd xl-srch.c

xl-dag.o: xl-dag.c
	charmc -seq -cwd xl-dag.c

###############################################################################
#
# charmxlat++ and charmfilter++
#
###############################################################################

XPHEADERS=xp-extn.h xp-t.h xp-ytab.h

CHARMXLATXX=xp-lexyy.o xp-process.o xp-table.o xp-ytab.o
charmxlat++: $(CHARMXLATXX) xp-space.o
	charmc -seq -cwd -o charmxlat++ $(CHARMXLATXX)
	charmc -seq -cwd -o charmfilter++ xp-space.o

xp-lexyy.o: xp-lexyy.c $(XPHEADERS)
	charmc -seq -cwd xp-lexyy.c

xp-process.o: xp-process.c $(XPHEADERS)
	charmc -seq -cwd xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	charmc -seq -cwd xp-table.c

xp-ytab.o: xp-ytab.c $(XPHEADERS)
	charmc -seq -cwd xp-ytab.c

xp-space.o: xp-space.c $(XPHEADERS)
	charmc -seq -cwd xp-space.c


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

