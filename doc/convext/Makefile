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
# Revision 1.8  2000-06-22 22:38:03  milind
# Minor fixes.
#
# Revision 1.7  2000/06/22 22:21:07  milind
# Added table of contents to the PDF file, and also cleaned up the clean
# target.
#
# Revision 1.6  2000/06/22 14:39:33  milind
# Made compatible with pdflatex as well.
#
# Revision 1.5  2000/02/24 07:06:29  milind
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
	make clean
	touch index.tex
	latex manual.tex
	makeindex -o index.tex manual.idx
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	make clean
	touch index.tex
	latex manual.tex
	latex2html -local_icons manual.tex

pdf:
	make clean
	touch index.tex
	latex manual.tex
	makeindex -o index.tex manual.idx
	pdflatex manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/convext.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/convext.manual.ps
	gzip /expand1/groupMosaic/distrib/convext.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/convext.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/convext.manual.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/convext.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/convext.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/convext.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/convext.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/convext.manual/*

all: ps html pdf install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl manual.out

veryclean: clean
	rm -rf *.ps *.pdf manual/

