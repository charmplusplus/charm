############################################################################
# RCS INFORMATION:
#
# 	$RCSfile$
# 	$Author$	$Locker$		$State$
#	$Revision$	$Date$
#
############################################################################
# DESCRIPTION:
#
############################################################################
# REVISION HISTORY:
#
# $Log$
# Revision 1.11  1999-07-27 21:04:39  jmwrght1
# Filled in some of the CCS functions.  Changed Makefile and manual.tex to
# generate the index.
#
# Revision 1.10  1999/07/15 23:24:06  jmwrght1
# Added the chapter on CCS to the Converse manual.  At this point, it is
# basically copied verbatim from the html file in the tech notes function.
# I need to fix some of the tex code (still learning this stuff) and elaborate
# on the explanations of various things.
#
# Revision 1.9  1996/11/18 16:55:50  milind
# Made changes to the scheduler section to reflect the semantics of
# CsdScheduler(0). Also, changed the Makefile to have targets ps, html,
# install and all.
#
# Revision 1.8  1996/06/14 21:38:25  jyelon
# This thing doesn't work at all.  Restructured.
#
# Revision 1.7  1995/11/15 18:01:22  brunner
# Added cpvmacros.tex to Makefile so make co works.
#
# Revision 1.6  1995/11/01  22:04:21  milind
# Added automatic generation of index.
#
# Revision 1.5  1995/10/30  16:06:49  milind
# Added one more latex stage to remove undefined references.
#
# Revision 1.4  1995/10/10  21:40:41  jyelon
# *** empty log message ***
#
# Revision 1.3  1995/09/21  21:42:21  milind
# Added latex2html
#
# Revision 1.2  1995/09/21  15:37:33  milind
# Added other tex files
#
# Revision 1.2  1995/09/21  15:37:33  milind
# Added other tex files
#
# Revision 1.1  1995/09/20  16:34:32  brunner
# Initial revision
#
#
############################################################################

TEXFILES=cmi.tex cpm.tex cpvmacros.tex index.tex ldb.tex manual.tex\
msgmgr.tex queue.tex scheduler.tex simulator.tex threads.tex\
thsync.tex ccs.tex

ps:
	latex manual.tex
	latex manual.tex
	makeindex -o index.tex manual.idx
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	latex manual.tex
	latex2html manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/converse.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/converse.manual.ps
	gzip /expand1/groupMosaic/distrib/converse.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/converse.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/converse.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/converse.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/converse.manual/*

all: ps html install

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~ index.tex
	rm -rf manual

