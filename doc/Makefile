IDIR=../../doc
LNCMD=test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD=rm -f ./pplmanual.sty
all: ps pdf html

ps:
	if [ ! -d $(IDIR)/ps ] ; then mkdir $(IDIR)/ps ; fi
	( cd install; $(LNCMD); make ps )
	( cd converse; $(LNCMD); make ps )
	( cd convext; $(LNCMD); make ps )
	( cd charm++; $(LNCMD); make ps )
	( cd ck; $(LNCMD); make ps )
	( cd libraries; $(LNCMD); make ps )
	( cd f90charm; $(LNCMD); make ps )
	( cd fem; $(LNCMD); make ps )
	( cd ampi; $(LNCMD); make ps )
	( cd bluegene; $(LNCMD); make ps )

pdf:
	if [ ! -d $(IDIR)/pdf ] ; then mkdir $(IDIR)/pdf ; fi
	( cd install; $(LNCMD); make pdf )
	( cd converse; $(LNCMD); make pdf )
	( cd convext; $(LNCMD); make pdf )
	( cd charm++; $(LNCMD); make pdf )
	( cd ck; $(LNCMD); make pdf )
	( cd libraries; $(LNCMD); make pdf )
	( cd f90charm; $(LNCMD); make pdf )
	( cd fem; $(LNCMD); make pdf )
	( cd ampi; $(LNCMD); make pdf )
	( cd bluegene; $(LNCMD); make pdf )

html:
	if [ ! -d $(IDIR)/html ] ; then mkdir $(IDIR)/html ; fi
	( cd install; $(LNCMD); make html )
	( cd converse; $(LNCMD); make html )
	( cd convext; $(LNCMD); make html )
	( cd charm++; $(LNCMD); make html )
	( cd ck; $(LNCMD); make html )
	( cd libraries; $(LNCMD); make html )
	( cd f90charm; $(LNCMD); make html )
	( cd fem; $(LNCMD); make html )
	( cd ampi; $(LNCMD); make html )
	( cd bluegene; $(LNCMD); make html )

doc:
	( cd install; $(LNCMD); make doc )
	( cd converse; $(LNCMD); make doc )
	( cd convext; $(LNCMD); make doc )
	( cd charm++; $(LNCMD); make doc )
	( cd ck; $(LNCMD); make doc )
	( cd libraries; $(LNCMD); make doc )
	( cd f90charm; $(LNCMD); make doc )
	( cd fem; $(LNCMD); make doc )
	( cd ampi; $(LNCMD); make doc )
	( cd bluegene; $(LNCMD); make doc )

web:
	( cd install; $(LNCMD); make web )
	( cd converse; $(LNCMD); make web )
	( cd convext; $(LNCMD); make web )
	( cd charm++; $(LNCMD); make web )
	( cd ck; $(LNCMD); make web )
	( cd libraries; $(LNCMD); make web )
	( cd f90charm; $(LNCMD); make web )
	( cd fem; $(LNCMD); make web )
	( cd ampi; $(LNCMD); make web )
	( cd bluegene; $(LNCMD); make web )

clean:
	( cd install; $(RMCMD); make clean )
	( cd converse; $(RMCMD); make clean )
	( cd convext; $(RMCMD); make clean )
	( cd charm++; $(RMCMD); make clean )
	( cd ck; $(RMCMD); make clean )
	( cd libraries; $(RMCMD); make clean )
	( cd f90charm; $(RMCMD); make clean )
	( cd fem; $(RMCMD); make clean )
	( cd ampi; $(RMCMD); make clean )
	( cd bluegene; $(RMCMD); make clean )
