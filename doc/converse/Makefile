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
# Revision 1.5  1995-10-30 16:06:49  milind
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

TEXFILES=manual.tex threads.tex queue.tex scheduler.tex ldb.tex cmi.tex thsync.tex msgmgr.tex

all: manual.ps manual/manual.html

ps-preview: manual.ps
	ghostview manual.ps

html-preview: manual/manual.html
	netscape manual/manual.html

manual/manual.html: $(TEXFILES)
	latex2html manual.tex

manual.ps: manual.dvi
	dvips -o manual.ps manual.dvi

manual.dvi: $(TEXFILES)
	latex manual.tex
	latex manual.tex

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~
	rm -rf manual
