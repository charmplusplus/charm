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
# Revision 1.2  1997-07-26 16:04:45  jyelon
# Added stuff.
#
#
############################################################################

TEXFILES=cmi.tex cpm.tex cpvmacros.tex index.tex ldb.tex manual.tex\
msgmgr.tex queue.tex scheduler.tex simulator.tex threads.tex\
thsync.tex pthreads.tex


ps:
	latex manual.tex
	latex manual.tex
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

