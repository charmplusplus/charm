IDIR=../../doc
LNCMD=test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD=rm -f ./pplmanual.sty
all: ps pdf html

ps:
	if [ ! -d $(IDIR)/ps ] ; then mkdir $(IDIR)/ps ; fi
	( cd install; $(LNCMD); /usr/ccs/bin/make ps )
	( cd converse; $(LNCMD); /usr/ccs/bin/make ps )
	( cd convext; $(LNCMD); /usr/ccs/bin/make ps )
	( cd charm++; $(LNCMD); /usr/ccs/bin/make ps )
	( cd ck; $(LNCMD); /usr/ccs/bin/make ps )
	( cd libraries; $(LNCMD); /usr/ccs/bin/make ps )
	( cd f90charm; $(LNCMD); /usr/ccs/bin/make ps )
	( cd fem; $(LNCMD); /usr/ccs/bin/make ps )
	( cd ampi; $(LNCMD); /usr/ccs/bin/make ps )
	( cd bluegene; $(LNCMD); /usr/ccs/bin/make ps )
	( cd mblock; $(LNCMD); /usr/ccs/bin/make ps )

pdf:
	if [ ! -d $(IDIR)/pdf ] ; then mkdir $(IDIR)/pdf ; fi
	( cd install; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd converse; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd convext; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd charm++; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd ck; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd libraries; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd f90charm; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd fem; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd ampi; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd bluegene; $(LNCMD); /usr/ccs/bin/make pdf )
	( cd mblock; $(LNCMD); /usr/ccs/bin/make pdf )

html:
	if [ ! -d $(IDIR)/html ] ; then mkdir $(IDIR)/html ; fi
	( cd install; $(LNCMD); /usr/ccs/bin/make html )
	( cd converse; $(LNCMD); /usr/ccs/bin/make html )
	( cd convext; $(LNCMD); /usr/ccs/bin/make html )
	( cd charm++; $(LNCMD); /usr/ccs/bin/make html )
	( cd ck; $(LNCMD); /usr/ccs/bin/make html )
	( cd libraries; $(LNCMD); /usr/ccs/bin/make html )
	( cd f90charm; $(LNCMD); /usr/ccs/bin/make html )
	( cd fem; $(LNCMD); /usr/ccs/bin/make html )
	( cd ampi; $(LNCMD); /usr/ccs/bin/make html )
	( cd bluegene; $(LNCMD); /usr/ccs/bin/make html )
	( cd mblock; $(LNCMD); /usr/ccs/bin/make html )

doc:
	( cd install; $(LNCMD); /usr/ccs/bin/make doc )
	( cd converse; $(LNCMD); /usr/ccs/bin/make doc )
	( cd convext; $(LNCMD); /usr/ccs/bin/make doc )
	( cd charm++; $(LNCMD); /usr/ccs/bin/make doc )
	( cd ck; $(LNCMD); /usr/ccs/bin/make doc )
	( cd libraries; $(LNCMD); /usr/ccs/bin/make doc )
	( cd f90charm; $(LNCMD); /usr/ccs/bin/make doc )
	( cd fem; $(LNCMD); /usr/ccs/bin/make doc )
	( cd ampi; $(LNCMD); /usr/ccs/bin/make doc )
	( cd bluegene; $(LNCMD); /usr/ccs/bin/make doc )
	( cd mblock; $(LNCMD); /usr/ccs/bin/make doc )

web:
	( cd install; $(LNCMD); /usr/ccs/bin/make web )
	( cd converse; $(LNCMD); /usr/ccs/bin/make web )
	( cd convext; $(LNCMD); /usr/ccs/bin/make web )
	( cd charm++; $(LNCMD); /usr/ccs/bin/make web )
	( cd ck; $(LNCMD); /usr/ccs/bin/make web )
	( cd libraries; $(LNCMD); /usr/ccs/bin/make web )
	( cd f90charm; $(LNCMD); /usr/ccs/bin/make web )
	( cd fem; $(LNCMD); /usr/ccs/bin/make web )
	( cd ampi; $(LNCMD); /usr/ccs/bin/make web )
	( cd bluegene; $(LNCMD); /usr/ccs/bin/make web )
	( cd mblock; $(LNCMD); /usr/ccs/bin/make web )

clean:
	( cd install; $(RMCMD); /usr/ccs/bin/make clean )
	( cd converse; $(RMCMD); /usr/ccs/bin/make clean )
	( cd convext; $(RMCMD); /usr/ccs/bin/make clean )
	( cd charm++; $(RMCMD); /usr/ccs/bin/make clean )
	( cd ck; $(RMCMD); /usr/ccs/bin/make clean )
	( cd libraries; $(RMCMD); /usr/ccs/bin/make clean )
	( cd f90charm; $(RMCMD); /usr/ccs/bin/make clean )
	( cd fem; $(RMCMD); /usr/ccs/bin/make clean )
	( cd ampi; $(RMCMD); /usr/ccs/bin/make clean )
	( cd bluegene; $(RMCMD); /usr/ccs/bin/make clean )
	( cd mblock; $(RMCMD); /usr/ccs/bin/make clean )
