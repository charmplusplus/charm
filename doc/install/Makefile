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
# Revision 1.2  1995-10-31 22:33:53  milind
# Changed to reflect the new structure of the manual.
#
# Revision 1.1  1995/09/20  16:26:49  brunner
# Initial revision
#
# Revision 1.2  1995/09/20  16:19:51  brunner
#
############################################################################

all: manual.ps manual/manual.html

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
