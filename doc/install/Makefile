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
# Revision 1.4  1996-11-14 18:41:13  milind
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

ps: manual.ps

html: manual/manual.html

install:
	cp manual.ps /expand1/groupMosaic/distrib/install.manual.ps
	rm -f /expand1/groupMosaic/distrib/install.manual.ps.gz
	gzip /expand1/groupMosaic/distrib/install.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/install.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/install.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/install.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/install.manual/*

all: ps html install

TEXFILES=manual.tex install.tex compile.tex run.tex

manual.ps:	manual.dvi
	dvips -o manual.ps manual.dvi

manual.dvi:	$(TEXFILES)
	latex manual
	latex manual

manual/manual.html: $(TEXFILES)
	latex2html manual.tex

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~
	rm -rf manual
