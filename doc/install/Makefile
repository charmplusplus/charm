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
# Revision 1.1  1995-09-20 16:26:49  brunner
# Initial revision
#
# Revision 1.2  1995/09/20  16:19:51  brunner
#
############################################################################

TEXFILES=install.tex

install.ps:	install.dvi
	dvips -o install.ps install.dvi

install.dvi:	$(TEXFILES)
	latex install
	latex install

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~
