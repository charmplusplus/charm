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
# This is a makefile.  It is used when you type ``make''.
#
############################################################################
# REVISION HISTORY:
#
# $Log$
# Revision 1.5  2000-02-24 07:06:29  milind
# Added new Crn functionality descriptions. And added an index to the
# manual.
#
# Revision 1.4  2000/02/20 17:36:52  milind
# Added -local_icons option to latex2html.
#
# Revision 1.3  1997/08/18 18:25:55  jyelon
# *** empty log message ***
#
# Revision 1.2  1997/07/26 16:04:45  jyelon
# Added stuff.
#
#
############################################################################

TEXFILES=cmi.tex cpm.tex cpvmacros.tex index.tex ldb.tex manual.tex\
msgmgr.tex queue.tex scheduler.tex simulator.tex threads.tex\
thsync.tex pthreads.tex


ps:
	touch index.tex
	latex manual.tex
	makeindex -o index.tex manual.idx
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	latex manual.tex
	latex2html -local_icons manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/convext.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/convext.manual.ps
	gzip /expand1/groupMosaic/distrib/convext.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/convext.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/convext.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/convext.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/convext.manual/*

all: ps html install

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~ index.tex
	rm -rf manual

