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
# Revision 1.8  1996-06-14 21:38:25  jyelon
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
thsync.tex

all:
	rm -f index.tex
	echo > index.tex
	latex manual.tex
	#texindex manual
	#mv manual.ind index.tex
	dvips -o manual.ps manual.dvi
	cp manual.ps /charm/manuals-comp/converse.manual/

pre: all
	ghostview manual.ps

html: all
	latex2html manual.tex

co:
	co Makefile $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc *~ index.tex
	rm -rf manual

