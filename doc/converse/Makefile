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
# Revision 1.2  1995-09-21 15:37:33  milind
# Added other tex files
#
# Revision 1.1  1995/09/20  16:34:32  brunner
# Initial revision
#
#
############################################################################

TEXFILES=manual.tex threads.tex queue.tex scheduler.tex ldb.tex cmi.tex

manual.ps:	manual.dvi
	dvips -o manual.ps manual.dvi

manual.dvi:	$(TEXFILES)
	latex manual
	latex manual

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~
