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

all: charm++ smsg pvm sdag

charm: converse charmxlat charmlibs libcharm

charm++: converse charmxlat charmlibs charmxlat++ charmfilter++ charmxi libcharm libck-cplus-y.a libck-defmain-cpp.a

smsg: converse libsimplemsg.a

pvm: converse libpvmc.a

sdag: charm++ sdagx libsdag.a

translators: sdagx charmxlat charmfilter charmxlat++ charmfilter++ charmxi conv-cpm

test-all: all
	(cd ../pgms ; make test-all)

test-converse: converse
	(cd ../pgms ; make test-converse)

test-charm: charm
	(cd ../pgms ; make test-charm)

test-charm++: charm++
	(cd ../pgms ; make test-charm++)

CHARMC=../bin/charmc $(CHARMOPTS) $(OPTS)

SEQCHARMC=../bin/charmc -seq $(CHARMOPTS)


###############################################################################
#
# The basics (bin, lib, include, headers, charmc, conv-mach.csh)
#
###############################################################################

ALLHEADERS=charm++.h charm.h charm++-inc.h charm-inc.h dagger.h cpthreads.h converse.h conv-mach.h simplemsg.h trace.h generic_redn.h pvmc.h pvm3.h sdag.h CDep.h CCounter.h CMsgBuffer.h CWhenTrigger.h TList.h # pglib.h

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
	@(cd .. ; basename `pwd`) 			> .vdir
	@(cd .. ; basename `pwd` | sed -e 's@-.*@@') 	> .gdir
	@echo
	@echo 'Gathering source code.'
	@echo
	find . -type l -exec rm {} \;
	rm -rf QuickThreads
	rm -rf ../bin ; mkdir ../bin
	rm -rf ../lib ; mkdir ../lib
	rm -rf ../src ; mkdir ../src
	rm -rf ../pgms
	../../src/Common/scripts/gatherflat ../../src/Common/scripts .
	./gatherflat ../../src/Common/conv-core        .
	./gatherflat ../../src/Common/conv-ldb         .
	./gatherflat ../../src/Common/ck-core          .
	./gatherflat ../../src/Common/ck-perf          .
	./gatherflat ../../src/Common/langs/simplemsg  .
	./gatherflat ../../src/Common/langs/pvmc       .
	./gatherflat ../../src/Common/langs/sdag/xlator       .
	./gatherflat ../../src/Common/langs/sdag/runtime      .
	./gatherflat ../../src/Common/lib-acc          .
	./gatherflat ../../src/Common/lib-misc         .
	./gatherflat ../../src/Common/lib-redn         .
	./gatherflat ../../src/Common/xlat             .
	./gatherflat ../../src/Common/xlat++           .
	./gatherflat ../../src/Common/xlat-i           .
	./gatherflat ../../src/Common/xlatcpm          .
	./gathertree ../../src/QuickThreads QuickThreads
	./gatherflat ../../src/Common.`cat .gdir`      .
	./gatherflat ../../src/`cat .vdir`             .
	./gathertree ../../src/Common/langs   ../src/langs
	./gathertree ../../src/Common/xlat    ../src/xlat
	./gathertree ../../src/Common/xlat++  ../src/xlat++
	./gathertree ../../src/Common/xlat-i  ../src/xlat-i
	./gathertree ../../src/Common/xlatcpm ../src/xlatcpm
	./gathertree ../../pgms ../pgms
	ln -s ../../src/Common/scripts/Makefile ../src
	@echo
	@echo
	@rm -f .vdir .gdir
	touch dirs+sources


###############################################################################
#
# Converse Libraries
#
###############################################################################

CVHEADERS=converse.h conv-mach.h conv-mach.csh

TRACELIBS=libtrace-none.a libtrace-summary.a libtrace-projections.a

CVLIBS=libconv-core.a libconv-cplus-n.a libconv-cplus-y.a libldb-rand.o $(TRACELIBS)

LIBCONV_CORE=threads.o convcore.o conv-conds.o spantree.o queueing.o fifo.o msgmgr.o memory.o cpm.o cpthreads.o futures.o cldb.o

converse: basics QuickThreads/libqt.a $(CVLIBS) conv-host-if-needed conv-cpm

conv-host-if-needed:
	if [ -f conv-host.c ] ; then make conv-host OPTS='$(OPTS)' ; fi

QuickThreads/libqt.a:
	csh -fc 'source conv-mach.csh;cd QuickThreads;./configure $$CMK_QT'
	cd QuickThreads ; make qt
	cp QuickThreads/libqt.a ../lib

conv-host: conv-host.c
	../bin/charmc $(OPTS) -c -seq -DSILENT conv-host.c
	../bin/charmc $(OPTS) -cp ../bin -seq -o conv-host-silent conv-host.o
	../bin/charmc $(OPTS) -c -seq -DNOTIFY conv-host.c
	../bin/charmc $(OPTS) -cp ../bin -seq -o conv-host-notify conv-host.o
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

cldb.o: cldb.c $(CVHEADERS)
	$(CHARMC) cldb.c

libldb-rand.o: cldb.rand.c $(CVHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ cldb.rand.c


###############################################################################
#
# Charm Libraries
#
###############################################################################

CKHEADERS=charm.h trace.h $(CVHEADERS)

CK_LIBS_CORE=libck-core.a

CK_LIBS_LDB=

CK_LIBS_CPLUS=libck-cplus-n.a libck-defmain-std.a

CK_LIBS=$(CK_LIBS_CORE) $(CK_LIBS_LDB) $(CK_LIBS_CPLUS)

LIBCK_CORE=common.o globals.o init.o register.o boc.o condsend.o qd.o vid.o rdonly.o stat.o wrtone.o acc.o mono.o tbl.o memstuff.o ck.o main.o

charmlibs: converse $(CK_LIBS)

libck-core.a: $(LIBCK_CORE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCK_CORE)

LIBTRACE_NONE=trace-none.o
libtrace-none.a: $(LIBTRACE_NONE)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_NONE)

LIBTRACE_SUMMARY=trace-summary.o traceio.o
libtrace-summary.a: $(LIBTRACE_SUMMARY)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_SUMMARY)

LIBTRACE_PROJ=trace-projections.o traceio.o
libtrace-projections.a: $(LIBTRACE_PROJ)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBTRACE_PROJ)

libldb-acwn.o: ldb.acwn.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.acwn.p

libldb-mngr.o: ldb.mngr.p ldb.h $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ ldb.mngr.p

libck-cplus-n.a: dummy.cplus.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ dummy.cplus.c

libck-defmain-std.a: defmain-std.c $(CKHEADERS)
	$(CHARMC) -cp ../lib/ -o $@ defmain-std.c

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

condsend.o: condsend.c $(CKHEADERS)
	$(CHARMC) -o condsend.o condsend.c

qd.o: qd.c $(CKHEADERS)
	$(CHARMC) -o qd.o qd.c

vid.o: vid.c $(CKHEADERS)
	$(CHARMC) -o vid.o vid.c

stat.o: stat.c $(CKHEADERS)
	$(CHARMC) -o stat.o stat.c

rdonly.o: rdonly.c $(CKHEADERS)
	$(CHARMC) -o rdonly.o rdonly.c

acc.o: acc.c $(CKHEADERS)
	$(CHARMC) -o acc.o acc.c

mono.o: mono.c $(CKHEADERS)
	$(CHARMC) -o mono.o mono.c

tbl.o: tbl.c $(CKHEADERS)
	$(CHARMC) -o tbl.o tbl.c

wrtone.o: wrtone.c $(CKHEADERS)
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
# C++ code for the charm++ runtime library.
#
###############################################################################

ckfutures.o: ckfutures.C ckfutures.ci charm++.h $(CKHEADERS)
	$(CHARMC) ckfutures.ci
	$(CHARMC) -c ckfutures.C

cplus.o: cplus.C charm++.h $(CKHEADERS)
	$(CHARMC) -o cplus.o cplus.C

libck-cplus-y.a: cplus.o ckfutures.o 
	$(CHARMC) -cp ../lib/ -o $@ cplus.C ckfutures.o

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
	 dminredn.o dmaxredn.o dsumredn.o dprodredn.o


libcharm: charmxlat libcharm.a

libcharm.a: $(LIBCHARM)
	$(CHARMC) -cp ../lib/ -o $@ $(LIBCHARM)

pglib.o: pglib.p pglib.h pg.int $(CKHEADERS)
	$(CHARMC) pglib.p

cache.o: cache.p cache.int $(CKHEADERS)
	$(CHARMC) cache.p

dag.o: dag.c $(CKHEADERS)
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
#
###############################################################################

CHARMXLAT=xl-main.o xl-yacc.tab.o xl-sym.o xl-str.o xl-lex.o xl-lexsp.o xl-rdtok.o xl-parse.o xl-outh.o xl-srch.o xl-dag.o

charmxlat charmfilter: $(CHARMXLAT) xl-sfspec.o
	$(SEQCHARMC) -o charmxlat   -cp ../bin/ $(CHARMXLAT)
	$(SEQCHARMC) -o charmfilter -cp ../bin/ xl-sfspec.o

xl-main.o: xl-main.c
	$(SEQCHARMC) xl-main.c

xl-sym.o: xl-sym.c xl-sym.h
	$(SEQCHARMC) xl-sym.c

xl-str.o: xl-str.c
	$(SEQCHARMC) xl-str.c

xl-lex.o: xl-lex.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-lex.c

xl-rdtok.o: xl-rdtok.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-rdtok.c

xl-parse.o: xl-parse.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-parse.c

xl-outh.o: xl-outh.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-outh.c

xl-srch.o: xl-srch.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-srch.c

xl-dag.o: xl-dag.c
	$(SEQCHARMC) xl-dag.c

xl-yacc.tab.o: xl-yacc.tab.c xl-lex.h xl-sym.h
	$(SEQCHARMC) xl-yacc.tab.c

xl-lexsp.o: xl-lexsp.c	
	$(SEQCHARMC) xl-lexsp.c

xl-sfspec.o: xl-sfspec.c
	$(SEQCHARMC) xl-sfspec.c

###############################################################################
#
# charmxlat++ and charmfilter++
#
###############################################################################

XPHEADERS=xp-extn.h xp-t.tab.c

CHARMXLATXX=xp-process.o xp-table.o xp-t.tab.o xp-lexer.o

charmxlat++: $(CHARMXLATXX) xp-sfilter.o
	$(SEQCHARMC) -o charmxlat++   -cp ../bin/ $(CHARMXLATXX)
	$(SEQCHARMC) -o charmfilter++ -cp ../bin/ xp-sfilter.o

xp-process.o: xp-process.c $(XPHEADERS)
	$(SEQCHARMC) xp-process.c

xp-table.o: xp-table.c $(XPHEADERS)
	$(SEQCHARMC) xp-table.c

xp-t.tab.o: xp-t.tab.c $(XPHEADERS)
	$(SEQCHARMC) -DYYDEBUG xp-t.tab.c

xp-lexer.o: xp-lexer.c $(XPHEADERS)
	$(SEQCHARMC) xp-lexer.c

xp-sfilter.o: xp-sfilter.c $(XPHEADERS)
	$(SEQCHARMC) xp-sfilter.c

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

XIHEADERS=xi-symbol.h xi-parse.tab.C
CHARMXI=xi-main.o xi-symbol.o xi-generate.o xi-parse.tab.o xi-scan.o

charmxi: $(CHARMXI)
	$(SEQCHARMC) -language c++ -cp ../bin/ -o charmxi $(CHARMXI)

xi-main.o: xi-main.C $(XIHEADERS)
	$(SEQCHARMC) xi-main.C

xi-symbol.o: xi-symbol.C $(XIHEADERS)
	$(SEQCHARMC) xi-symbol.C

xi-generate.o: xi-generate.C $(XIHEADERS)
	$(SEQCHARMC) xi-generate.C

xi-parse.tab.o: xi-parse.tab.C $(XIHEADERS)
	$(SEQCHARMC) xi-parse.tab.C

xi-scan.o: xi-scan.C $(XIHEADERS)
	$(SEQCHARMC) xi-scan.C

###############################################################################
#
# Simple Messaging
#
###############################################################################

libsimplemsg.a: converse simplemsg.c simplemsg.h
	$(CHARMC) -cp ../lib/ -o $@ simplemsg.c

###############################################################################
#
# PVM
#
###############################################################################

PVMC=pvmc_buf.o pvmc_comm.o pvmc_conv.o pvmc_main.o pvmc_pack.o pvmc_groups.o

libpvmc.a: converse $(PVMC)
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

trans.c: trans.l
	flex -otrans.c trans.l

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
