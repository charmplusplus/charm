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
# Revision 1.9  2000-06-22 22:38:05  milind
# Minor fixes.
#
# Revision 1.8  2000/06/22 22:21:08  milind
# Added table of contents to the PDF file, and also cleaned up the clean
# target.
#
# Revision 1.7  2000/06/22 15:04:53  milind
# Made compatible with pdflatex as well.
#
# Revision 1.6  1999/07/09 18:18:39  milind
# Updated for new release 5.0.
#
# Revision 1.5  1997/08/18 18:37:34  jyelon
# *** empty log message ***
#
# Revision 1.4  1996/11/14 18:41:13  milind
# Modified to reflect the changes made to the nodesfile format.
# Also, changed the Makefile to install into proper directories.
#
# Revision 1.3  1995/11/16 19:05:03  milind
# *** empty log message ***
#
# Revision 1.2  1995/10/31  22:33:53  milind
# Changed to reflect the new structure of the manual.
#
# Revision 1.1  1995/09/20  16:26:49  brunner
# Initial revision
#
# Revision 1.2  1995/09/20  16:19:51  brunner
#
############################################################################

TEXFILES=manual.tex install.tex compile.tex run.tex

ps:
	make clean
	latex manual.tex
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	make clean
	latex manual.tex
	latex2html -local_icons manual.tex

pdf:
	make clean
	pdflatex manual.tex
	pdflatex manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/install.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/install.manual.ps
	gzip /expand1/groupMosaic/distrib/install.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/install.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/install.manual.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/install.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/install.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/install.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/install.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/install.manual/*

all: ps pdf html install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl

veryclean: clean
	rm -rf *.ps *.pdf manual/

