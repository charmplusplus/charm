IDIR=../../doc
LNCMD=ln -f -s ../pplmanual.sty .
RMCMD=rm -f ./pplmanual.sty
all: ps pdf html

ps:
	if [ ! -d $(IDIR)/ps ] ; then mkdir $(IDIR)/ps ; fi
	( cd install; $(LNCMD); make ps )
	( cd converse; $(LNCMD); make ps )
	( cd convext; $(LNCMD); make ps )
	( cd charm++; $(LNCMD); make ps )
	( cd cldb; $(LNCMD); make ps )
	( cd ck; $(LNCMD); make ps )
	( cd libraries; $(LNCMD); make ps )
	( cd f90charm; $(LNCMD); make ps )
	( cd fem; $(LNCMD); make ps )
	( cd ampi; $(LNCMD); make ps )

pdf:
	if [ ! -d $(IDIR)/pdf ] ; then mkdir $(IDIR)/pdf ; fi
	( cd install; $(LNCMD); make pdf )
	( cd converse; $(LNCMD); make pdf )
	( cd convext; $(LNCMD); make pdf )
	( cd charm++; $(LNCMD); make pdf )
	( cd cldb; $(LNCMD); make pdf )
	( cd ck; $(LNCMD); make pdf )
	( cd libraries; $(LNCMD); make pdf )
	( cd f90charm; $(LNCMD); make pdf )
	( cd fem; $(LNCMD); make pdf )
	( cd ampi; $(LNCMD); make pdf )

html:
	if [ ! -d $(IDIR)/html ] ; then mkdir $(IDIR)/html ; fi
	( cd install; $(LNCMD); make html )
	( cd converse; $(LNCMD); make html )
	( cd convext; $(LNCMD); make html )
	( cd charm++; $(LNCMD); make html )
	( cd cldb; $(LNCMD); make html )
	( cd ck; $(LNCMD); make html )
	( cd libraries; $(LNCMD); make html )
	( cd f90charm; $(LNCMD); make html )
	( cd fem; $(LNCMD); make html )
	( cd ampi; $(LNCMD); make html )

clean:
	( cd install; $(RMCMD); make clean )
	( cd converse; $(RMCMD); make clean )
	( cd convext; $(RMCMD); make clean )
	( cd charm++; $(RMCMD); make clean )
	( cd cldb; $(RMCMD); make clean )
	( cd ck; $(RMCMD); make clean )
	( cd libraries; $(RMCMD); make clean )
	( cd f90charm; $(RMCMD); make clean )
	( cd fem; $(RMCMD); make clean )
	( cd ampi; $(RMCMD); make clean )
