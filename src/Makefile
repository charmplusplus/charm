#############################################################################
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

SHELL=/bin/sh

all: charm++
#	@echo 'try one of:'
#	@echo '  make converse'
#	@echo '  make charm'
#	@echo '  make charm++'
#	@echo '  make test-converse'
#	@echo '  make test-charm'
#	@echo '  make test-charm++'

charm: converse charmxlat charmlibs libcharm

charm++: converse charmxlat charmlibs charmxlat++ charmfilter++ charmxi libcharm libck-cplus-y.a libck-defmain-cpp.a

translators: charmxlat charmfilter charmxlat++ charmfilter++ charmxi conv-cpm

test-converse: converse
	(cd ../pgms ; make test-converse)

test-charm: charm
	(cd ../pgms ; make test-charm)

test-charm++: charm++
	(cd ../pgms ; make test-charm++)

CHARMC=../bin/charmc $(CHARMOPTS) $(OPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.csh)
#
###############################################################################

ALLHEADERS=acc.h c++interface.h chare.h ckdefs.h common.h communication.h condsend.h const.h cpthreads.h dagger.h dtable.h env_macros.h globals.h mono.h msg_macros.h prio_macros.h qd.h stat.h sys_macros.h table.h tbl.h trans_decls.h trans_defs.h trans_externs.h user_macros.h vid.h wrtone.h converse.h conv-mach.h internal.h quiescence.h simplemsg.h trace.h dag.h generic_redn.h charmFuturesInclude.h # pglib.h

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
	@echo 'Creating directory tree '`cat .vdir`'/...'
	@echo
	rm -rf ../bin ; mkdir ../bin
	rm -rf ../lib ; mkdir ../lib
	rm -rf ../src ; mkdir ../src
	mkdir ../src/xlat
	mkdir ../src/xlat++
	mkdir ../src/xlat-i
	mkdir ../src/xlatcpm
	@echo
	@echo 'Gathering source code in '`cat .vdir`'/tmp'
	@echo
	rm -f ../../bin ; ln -s `cat .vdir`/bin ../../bin
	find . -type l -exec rm {} \;
	find ../../src/Common -type f \! -name Entries \! -name Entries.Log \! -name Root \! -name Repository -exec ln -s {} . \;
	find ../../src/`cat .vdir` -type f \! -name Entries \! -name Entries.Log \! -name Root \! -name Repository -exec ln -s {} . \;
	if [ -d ../../src/Common.`cat .gdir` ] ; then find ../../src/Common.`cat .gdir` -type f \! -name Entries \! -name Entries.Log \! -name Root \! -name Repository -exec ln -s {} . \; ; fi
	@echo
	@echo 'Gathering quickthreads tree.'
	@echo
	rm -rf QuickThreads ../src/QuickThreads
	(cd ../../src ; find QuickThreads -type d -print) | sed -e 's@^@mkdir @' > tmp.sh
	sh tmp.sh ; cd ../src ; sh ../tmp/tmp.sh ; rm -f tmp.sh
	rt=`(cd ../../src ; pwd)` ; (cd ../../src ; find QuickThreads -type f -print) | awk '{ print "ln -s '$$rt'/"$$1" "$$1; }' > tmp.sh
	sh tmp.sh ; cd ../src ; sh ../tmp/tmp.sh ; rm -f tmp.sh
	@echo
	@echo 'Gathering test-programs in '`cat .vdir`/pgms
	@echo
	rm -rf ../pgms
	(cd ../.. ; find pgms -type d -print) | sed -e 's@^@mkdir ../@' | sh
	rt=`(cd ../.. ; pwd)` ; (cd ../.. ; find pgms -type f -print) | awk '{ print "ln -s '$$rt'/"$$1" ../"$$1; }' > tmp.sh
	sh tmp.sh ; rm -f tmp.sh
	@echo
	@echo 'Gathering translator source in '`cat .vdir`/src
	@echo
	ln -s ../../src/Common/xlat/xl-*           ../src/xlat
	ln -s ../../src/Common/xlat++/xp-*         ../src/xlat++
	ln -s ../../src/Common/xlat-i/xi-*         ../src/xlat-i
	ln -s ../../src/Common/xlatcpm/conv-cpm.*  ../src/xlatcpm
	ln -s ../../src/Common/Makefile            ../src/xlat
	ln -s ../../src/Common/Makefile            ../src/xlat++
	ln -s ../../src/Common/Makefile            ../src/xlat-i
	ln -s ../../src/Common/Makefile            ../src/xlatcpm
	@echo
	@rm -f .vdir .gdir
	touch dirs+sources


###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h conv-mach.csh

CVLIBS=libconv-core.a libconv-cplus-n.a libconv-cplus-y.a

LIBCONV_CORE=threads.o convcore.o conv-conds.o spantree.o queueing.o fifo.o msgmgr.o memory.o cpm.o cpthreads.o futures.o

converse: basics QuickThreads/libqt.a $(CVLIBS) conv-host-if-needed conv-cpm

conv-host-if-needed:
	if [ -f conv-host.c ] ; then make conv-host OPTS='$(OPTS)' ; fi

QuickThreads/libqt.a:
	csh -fc 'source conv-mach.csh;cd QuickThreads;./configure $$CMK_QT'
	cd QuickThreads ; make qt
	cp QuickThreads/libqt.a ../lib

conv-host: conv-host.c
	../bin/charmc -cp ../bin -seq -DNOTIFY -o conv-host-notify conv-host.c
	../bin/charmc -cp ../bin -seq -DSILENT -o conv-host-silent conv-host.c
	cp conv-host-notify conv-host
	cp conv-host ../bin/conv-host

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

###############################################################################
#
# Charm Libraries
#
###############################################################################

CKHEADERS=chare.h const.h msg_macros.h prio_macros.h env_macros.h globals.h\
	  sys_macros.h communication.h trace.h $(CVHEADERS)

CK_LIBS_CORE=libck-core.a

CK_LIBS_TRACE=libck-trace-none.a libck-trace-summary.a libck-trace-projections.a

CK_LIBS_LDB=libck-ldb-rand.o libck-ldb-acwn.o libck-ldb-mngr.o # libck-ldb-priomngr.o libck-ldb-recv.o libck-ldb-tok.o libck-ldb-btok.o

CK_LIBS_CPLUS=libck-cplus-n.a libck-defmain-std.a

CK_LIBS=$(CK_LIBS_CORE) $(CK_LIBS_TRACE) $(CK_LIBS_LDB) $(CK_LIBS_CPLUS)

LIBCK_CORE=common.o globals.o init.o register.o boc.o condsend.o qd.o vid.o rdonly.o stat.o wrtone.o acc.o mono.o tbl.o memstuff.o ldbcfns.o ck.o main.o

charmlibs: converse $(CK_LIBS)

libck-core.a: $(LIBCK_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE)

LIBCK_TRACE_NONE=trace-none.o
libck-trace-none.a: $(LIBCK_TRACE_NONE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_TRACE_NONE)

LIBCK_TRACE_SUMMARY=trace-summary.o traceio.o
libck-trace-summary.a: $(LIBCK_TRACE_SUMMARY)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_TRACE_SUMMARY)

LIBCK_TRACE_PROJ=trace-projections.o traceio.o
libck-trace-projections.a: $(LIBCK_TRACE_PROJ)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_TRACE_PROJ)

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

libck-defmain-std.a: defmain-std.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ defmain-std.c

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

traceio.o : traceio.c trace.h $(CKHEADERS)
	$(CHARMC) -o traceio.o traceio.c

trace-none.o : trace-none.c trace.h $(CKHEADERS)
	$(CHARMC) -o trace-none.o trace-none.c

trace-summary.o : trace-summary.c trace.h $(CKHEADERS)
	$(CHARMC) -o trace-summary.o trace-summary.c

trace-record.o : trace-record.c trace.h $(CKHEADERS)
	$(CHARMC) -o trace-record.o trace-record.c

trace-replay.o : trace-replay.c trace.h $(CKHEADERS)
	$(CHARMC) -o trace-replay.o trace-replay.c

trace-projections.o : trace-projections.c trace.h $(CKHEADERS)
	$(CHARMC) -o trace-projections.o trace-projections.c

###############################################################################
#
# The new, lightweight C++ stub library.
#
###############################################################################

libck-cplus-y.a: cplus.C c++interface.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cplus.C

libck-defmain-cpp.a: defmain-cpp.C $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ defmain-cpp.C

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
	 dminredn.o dmaxredn.o dsumredn.o dprodredn.o \
	 charmFutures.o

libcharm: charmxlat libcharm.a

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

charmFutures.o: charmFutures.C
	$(CHARMC) charmFutures.ci
	$(CHARMC) -c charmFutures.C


###############################################################################
#
#
###############################################################################

CHARMXLAT=xl-main.o xl-yacc.tab.o xl-sym.o xl-str.o xl-lex.o xl-lexsp.o xl-rdtok.o xl-parse.o xl-outh.o xl-srch.o xl-dag.o

charmxlat charmfilter: $(CHARMXLAT) xl-sfspec.o
	$(CHARMC) -seq -o charmxlat   -cp ../bin/ $(CHARMXLAT)
	$(CHARMC) -seq -o charmfilter -cp ../bin/ xl-sfspec.o

xl-main.o: xl-main.c
	$(CHARMC) -seq xl-main.c

xl-sym.o: xl-sym.c xl-sym.h
	$(CHARMC) -seq xl-sym.c

xl-str.o: xl-str.c
	$(CHARMC) -seq xl-str.c

xl-lex.o: xl-lex.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-lex.c

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

xl-yacc.tab.o: xl-yacc.tab.c xl-lex.h xl-sym.h
	$(CHARMC) -seq xl-yacc.tab.c

xl-lexsp.o: xl-lexsp.c	
	$(CHARMC) -seq xl-lexsp.c

xl-sfspec.o: xl-sfspec.c
	$(CHARMC) -seq xl-sfspec.c

###############################################################################
#
# charmxlat++ and charmfilter++
#
###############################################################################

XPHEADERS=xp-extn.h xp-t.tab.c

CHARMXLATXX=xp-process.o xp-table.o xp-t.tab.o xp-lexer.o

charmxlat++: $(CHARMXLATXX) xp-sfilter.o
	$(CHARMC) -seq -o charmxlat++   -cp ../bin/ $(CHARMXLATXX)
	$(CHARMC) -seq -o charmfilter++ -cp ../bin/ xp-sfilter.o

xp-process.o: xp-process.c $(XPHEADERS)
	$(CHARMC) -seq xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	$(CHARMC) -seq xp-table.c

xp-t.tab.o: xp-t.tab.c $(XPHEADERS)
	$(CHARMC) -seq -DYYDEBUG xp-t.tab.c

xp-lexer.o: xp-lexer.c $(XPHEADERS)
	$(CHARMC) -seq xp-lexer.c

xp-sfilter.o: xp-sfilter.c $(XPHEADERS)
	$(CHARMC) -seq xp-sfilter.c

###############################################################################
#
# The CPM scanner
#
###############################################################################

conv-cpm: conv-cpm.o
	$(CHARMC) -seq -o conv-cpm -cp ../bin/ conv-cpm.o

conv-cpm.o: conv-cpm.c
	$(CHARMC) -seq conv-cpm.c

###############################################################################
#
# The interface translator
#
###############################################################################

XIHEADERS=xi-symbol.h xi-parse.tab.C
CHARMXI=xi-main.o xi-symbol.o xi-generate.o xi-parse.tab.o xi-scan.o

charmxi: $(CHARMXI)
	$(CHARMC) -seq -language c++ -cp ../bin/ -o charmxi $(CHARMXI)

xi-main.o: xi-main.C $(XIHEADERS)
	$(CHARMC) -seq xi-main.C

xi-symbol.o: xi-symbol.C $(XIHEADERS)
	$(CHARMC) -seq xi-symbol.C

xi-generate.o: xi-generate.C $(XIHEADERS)
	$(CHARMC) -seq xi-generate.C

xi-parse.tab.o: xi-parse.tab.C $(XIHEADERS)
	$(CHARMC) -seq xi-parse.tab.C

xi-scan.o: xi-scan.C $(XIHEADERS)
	$(CHARMC) -seq xi-scan.C

###############################################################################
#
# Little languages.
#
###############################################################################

libsimplemsg.a: simplemsg.c simplemsg.h
	$(CHARMC) -cp ../lib/ -o $@ simplemsg.c

###############################################################################
#
# Make clean
#
###############################################################################

clean:
	rm -f charmreplay charmxlat charmfilter charmxlat++ charmfilter++
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
