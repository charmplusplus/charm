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

all: charm++
#	@echo 'try one of:'
#	@echo '  make converse'
#	@echo '  make charm'
#	@echo '  make charm++'

charm: converse charmxlat charmlibs libcharm

charm++: converse charmxlat charmlibs charmxlat++ libcharm libck-cplus-y.a

OPTS=-g
CHARMC=../bin/charmc $(OPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.csh)
#
###############################################################################

ALLHEADERS=acc.h c++interface.h chare.h ckdefs.h common.h communication.h condsend.h const.h dagger.h dtable.h env_macros.h globals.h mono.h msg_macros.h prio_macros.h qd.h stat.h sys_macros.h table.h tbl.h trans_decls.h trans_defs.h trans_externs.h user_macros.h vid.h wrtone.h converse.h conv-mach.h internal.h quiescence.h performance.h dag.h generic_redn.h # pglib.h

ALLINTERFACES=GENERIC-ACC.int GENERIC-HIST.int GENERIC-REDN.int barrier_redn.int cache.int dacc_count.int dacc_hist.int dcountredn.int dmaxredn.int dminredn.int dprodredn.int dsumredn.int facc_count.int facc_hist.int fcountredn.int fmaxredn.int fminredn.int fprodredn.int fsumredn.int iacc_count.int iacc_hist.int icountredn.int imaxredn.int iminredn.int iprodredn.int isumredn.int pg.int

basics: dirs+sources $(ALLHEADERS) $(ALLINTERFACES) charmc conv-mach.csh
	rm -rf ../include
	mkdir ../include
	cp $(ALLHEADERS) ../include/
	cp $(ALLINTERFACES) ../include/
	rm -f ../bin/charmc ; cp charmc ../bin/
	chmod 755 ../bin/charmc
	cp conv-mach.csh ../include/
	touch basics

dirs+sources:
	@rm -f .vdir .gdir
	@(cd .. ; basename `pwd` > tmp/.vdir)
	@cat .vdir | sed -e 's@-.*@@' > .gdir
	@echo
	@echo 'Gathering source code in '`cat .vdir`'/tmp'
	@echo
	if [ ! -d ../bin ] ; then mkdir ../bin ; fi
	if [ ! -d ../lib ] ; then mkdir ../lib ; fi
	rm -f ../../bin ; ln -s `cat .vdir`/bin ../../bin
	find . -type l -exec rm {} \;
	find ../../src/Common -type f \! -name "*,v" -exec ln -s {} . \;
	find ../../src/`cat .vdir` -type f \! -name "*,v" -exec ln -s {} . \;
	if [ -d ../../src/Common.`cat .gdir` ] ; then find ../../src/Common.`cat .gdir` -type f \! -name "*,v" -exec ln -s {} . \; ; fi
	@echo
	@echo 'Gathering test-programs in '`cat .vdir`/pgms
	@echo
	rm -rf ../pgms
	(cd ../.. ; find pgms -type d -print) | sed -e 's@^@mkdir ../@' | sh
	rt=`(cd ../.. ; pwd)` ; (cd ../.. ; find pgms -type f -print) | awk '{ print "ln -s '$$rt'/"$$1" ../"$$1; }' | sh
	@rm -f .vdir .gdir
	touch dirs+sources



###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h 

CVLIBS=libconv-core.a $(CVLIBS_QS)

LIBCONV_CORE=convcore.o conv-conds.o machine.o spantree.o queueing.o fifo.o threads.o msgmgr.o memory.o

converse: basics $(CVLIBS) cmk_extras

libconv-core.a: $(LIBCONV_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCONV_CORE)

fifo.o: fifo.c $(CVHEADERS)
	$(CHARMC) fifo.c

spantree.o: spantree.c $(CVHEADERS)
	$(CHARMC) spantree.c

machine.o: machine.c $(CVHEADERS)
	$(CHARMC) machine.c

convcore.o: convcore.c $(CVHEADERS)
	$(CHARMC) convcore.c

queueing.o: queueing.c $(CVHEADERS)
	$(CHARMC) queueing.c

conv-conds.o: conv-conds.c $(CVHEADERS)
	$(CHARMC) conv-conds.c

threads.o: threads.c $(CVHEADERS)
	$(CHARMC) threads.c

msgmgr.o: msgmgr.c $(CVHEADERS)
	$(CHARMC) msgmgr.c

memory.o: memory.c $(CVHEADERS)
	$(CHARMC) memory.c

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
###############################################################################

CKHEADERS=chare.h const.h msg_macros.h prio_macros.h env_macros.h globals.h\
	  sys_macros.h communication.h performance.h $(CVHEADERS)

CK_LIBS_CORE=libck-core-notrace.a libck-core-summary.a libck-core-projections.a

CK_LIBS_LDB=libck-ldb-rand.o libck-ldb-acwn.o libck-ldb-mngr.o # libck-ldb-priomngr.o libck-ldb-recv.o libck-ldb-tok.o libck-ldb-btok.o

CK_LIBS_CPLUS=libck-cplus-n.a

CK_LIBS_MAIN=libck-defmain.o libck-unimain.o

CK_LIBS=$(CK_LIBS_CORE) $(CK_LIBS_LDB) $(CK_LIBS_CPLUS) $(CK_LIBS_MAIN)

LIBCK_CORE=common.o globals.o init.o register.o boc.o condsend.o qd.o vid.o rdonly.o stat.o wrtone.o acc.o mono.o tbl.o memstuff.o ldbcfns.o ck.o main.o

charmlibs: converse $(CK_LIBS)

LIBCK_CORE_NOTRACE=$(LIBCK_CORE) perf-none.o
libck-core-notrace.a: $(LIBCK_CORE_NOTRACE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE_NOTRACE)

LIBCK_CORE_SUMMARY=$(LIBCK_CORE) perf-sum.o
libck-core-summary.a: $(LIBCK_CORE_SUMMARY)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE_SUMMARY)

LIBCK_CORE_PROJ=$(LIBCK_CORE) perf-proj.o perfio.o
libck-core-projections.a: $(LIBCK_CORE_PROJ)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE_PROJ)

libck-ldb-acwn.o: ldb.acwn.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.acwn.p

libck-ldb-rand.o: ldb.rand.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.rand.p

libck-ldb-mngr.o: ldb.mngr.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.mngr.p

libck-ldb-priomngr.o: ldb.priomngr.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.priomngr.p

libck-ldb-tok.o: ldb.tok.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.tok.p

libck-ldb-btok.o: ldb.btok.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.tok.p -DBITVECTOR

libck-ldb-repl.o: ldb.repl.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.repl.p

libck-ldb-recv.o: ldb.recv.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.recv.p

libck-cplus-n.a: dummy.cplus.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ dummy.cplus.c

libck-defmain.o: default-main.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ default-main.c

libck-unimain.o: uni_user_main.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ uni_user_main.c

ldbcfns.o: ldbcfns.c $(CKHEADERS)
	$(CHARMC) -o ldbcfns.o ldbcfns.c

common.o: common.c $(CKHEADERS)
	$(CHARMC) -o common.o common.c

globals.o: globals.c $(CKHEADERS)
	$(CHARMC) -o globals.o globals.c

register.o: register.c $(CKHEADERS)
	$(CHARMC) -o register.o register.c

init.o: init.c $(CKHEADERS)
	$(CHARMC) -o init.o init.c

boc.o: boc.c $(CKHEADERS)
	$(CHARMC) -o boc.o boc.c

condsend.o: condsend.c condsend.h $(CKHEADERS)
	$(CHARMC) -o condsend.o condsend.c

qd.o: qd.c qd.h $(CKHEADERS)
	$(CHARMC) -o qd.o qd.c

vid.o: vid.c vid.h $(CKHEADERS)
	$(CHARMC) -o vid.o vid.c

stat.o: stat.c stat.h $(CKHEADERS)
	$(CHARMC) -o stat.o stat.c

rdonly.o: rdonly.c $(CKHEADERS)
	$(CHARMC) -o rdonly.o rdonly.c

acc.o: acc.c acc.h $(CKHEADERS)
	$(CHARMC) -o acc.o acc.c

mono.o: mono.c mono.h $(CKHEADERS)
	$(CHARMC) -o mono.o mono.c

tbl.o: tbl.c tbl.h $(CKHEADERS)
	$(CHARMC) -o tbl.o tbl.c

wrtone.o: wrtone.c wrtone.h $(CKHEADERS)
	$(CHARMC) -o wrtone.o wrtone.c

memstuff.o: memstuff.c $(CKHEADERS)
	$(CHARMC) -o memstuff.o memstuff.c

main.o: main.c $(CKHEADERS)
	$(CHARMC) -o main.o main.c

ck.o: ck.c $(CKHEADERS)
	$(CHARMC) -o ck.o ck.c

perfio.o : perfio.c performance.h $(CKHEADERS)
	$(CHARMC) -o perfio.o perfio.c

perf-none.o : perf-none.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-none.o perf-none.c

perf-sum.o : perf-sum.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-sum.o perf-sum.c

perf-recd.o : perf-recd.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-recd.o perf-recd.c

perf-repl.o : perf-repl.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-repl.o perf-repl.c

perf-proj.o : perf-proj.c performance.h $(CKHEADERS)
	$(CHARMC) -o perf-proj.o perf-proj.c

###############################################################################
#
# The new, lightweight C++ stub library.
#
###############################################################################

libck-cplus-y.a: cplus.C c++interface.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cplus.C

###############################################################################
#
# libcharm
#
###############################################################################

GENERIC_ACC_H=GENERIC-ACC.int GENERIC-ACC.p $(CKHEADERS)
GENERIC_HIST_H=GENERIC-HIST.int GENERIC-HIST.p $(CKHEADERS)
GENERIC_REDN_H=GENERIC-REDN.int GENERIC-REDN.p $(CKHEADERS)

LIBCHARM=cache.o dag.o barrier_redn.o pglib.o \
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

xl-main.o: xl-main.c
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

CHARMXLATXX=xp-lexyy.o xp-process.o xp-table.o xp-ytab.o xp-ycode.o

charmxlat++: basics $(CHARMXLATXX) xp-space.o
	$(CHARMC) -seq -o charmxlat++   -cp ../bin/ $(CHARMXLATXX)
	$(CHARMC) -seq -o charmfilter++ -cp ../bin/ xp-space.o

xp-lexyy.o: xp-lexyy.c $(XPHEADERS)
	$(CHARMC) -seq xp-lexyy.c

xp-process.o: xp-process.c $(XPHEADERS)
	$(CHARMC) -seq xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	$(CHARMC) -seq xp-table.c

xp-ycode.o: xp-ycode.c $(XPHEADERS)
	$(CHARMC) -seq xp-ycode.c

xp-ytab.o: xp-ytab.c $(XPHEADERS)
	$(CHARMC) -seq -DYYDEBUG xp-ytab.c

xp-space.o: xp-space.c $(XPHEADERS)
	$(CHARMC) -seq xp-space.c


###############################################################################
#
# Rules for Charm++ translator and spacefilter
#
###############################################################################

yacc++: xlat++yacc xlat++lex sfilter++lex
	touch xlat++yacc xlat++lex sfilter++lex

xlat++yacc: xp-t.y
	byacc -d -r xp-t.y
	cat y.tab.h > xp-ytab.h
	rm -f y.tab.h
	sed -e 's/, */,/g' -e 's/^ *//' < y.tab.c > xp-ytab.c
	rm -f y.tab.c
	cat y.code.c > xp-ycode.c
	rm -f y.code.c

xlat++lex: xp-t.l
	flex xp-t.l
	cat lex.yy.c > xp-lexyy.c
	rm -f lex.yy.c

sfilter++lex: xp-sfilter
	flex xp-sfilter
	cat lex.yy.c > xp-space.c
	rm -f lex.yy.c


###############################################################################
#
# Make clean
#
###############################################################################

clean: cmk_clean
	rm -f charmreplay charmxlat charmfilter charm++xlat charm++filter
	rm -f conv-host
	rm -f TAGS basics dirs+sources cmk_extras core
	rm -f core #*#
	rm -f core *.a
	rm -f core *.o
	rm -f core *~
	rm -rf ../bin
	rm -rf ../lib
	rm -rf ../include


