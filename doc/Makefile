IDIR=../../doc
LNCMD=test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD=rm -f ./pplmanual.sty

all: ps pdf html

ps:
	if [ ! -d $(IDIR)/ps ] ; then mkdir $(IDIR)/ps ; fi
	( cd install; $(LNCMD); $(MAKE) ps )
	( cd converse; $(LNCMD); $(MAKE) ps )
	( cd convext; $(LNCMD); $(MAKE) ps )
	( cd charm++; $(LNCMD); $(MAKE) ps )
	( cd ck; $(LNCMD); $(MAKE) ps )
	( cd libraries; $(LNCMD); $(MAKE) ps )
	( cd f90charm; $(LNCMD); $(MAKE) ps )
	( cd fem; $(LNCMD); $(MAKE) ps )
	( cd ampi; $(LNCMD); $(MAKE) ps )
	( cd bluegene; $(LNCMD); $(MAKE) ps )
	( cd mblock; $(LNCMD); $(MAKE) ps )

pdf:
	if [ ! -d $(IDIR)/pdf ] ; then mkdir $(IDIR)/pdf ; fi
	( cd install; $(LNCMD); $(MAKE) pdf )
	( cd converse; $(LNCMD); $(MAKE) pdf )
	( cd convext; $(LNCMD); $(MAKE) pdf )
	( cd charm++; $(LNCMD); $(MAKE) pdf )
	( cd ck; $(LNCMD); $(MAKE) pdf )
	( cd libraries; $(LNCMD); $(MAKE) pdf )
	( cd f90charm; $(LNCMD); $(MAKE) pdf )
	( cd fem; $(LNCMD); $(MAKE) pdf )
	( cd ampi; $(LNCMD); $(MAKE) pdf )
	( cd bluegene; $(LNCMD); $(MAKE) pdf )
	( cd mblock; $(LNCMD); $(MAKE) pdf )

html:
	if [ ! -d $(IDIR)/html ] ; then mkdir $(IDIR)/html ; fi
	( cd install; $(LNCMD); $(MAKE) html )
	( cd converse; $(LNCMD); $(MAKE) html )
	( cd convext; $(LNCMD); $(MAKE) html )
	( cd charm++; $(LNCMD); $(MAKE) html )
	( cd ck; $(LNCMD); $(MAKE) html )
	( cd libraries; $(LNCMD); $(MAKE) html )
	( cd f90charm; $(LNCMD); $(MAKE) html )
	( cd fem; $(LNCMD); $(MAKE) html )
	( cd ampi; $(LNCMD); $(MAKE) html )
	( cd bluegene; $(LNCMD); $(MAKE) html )
	( cd mblock; $(LNCMD); $(MAKE) html )

doc:
	( cd install; $(LNCMD); $(MAKE) doc )
	( cd converse; $(LNCMD); $(MAKE) doc )
	( cd convext; $(LNCMD); $(MAKE) doc )
	( cd charm++; $(LNCMD); $(MAKE) doc )
	( cd ck; $(LNCMD); $(MAKE) doc )
	( cd libraries; $(LNCMD); $(MAKE) doc )
	( cd f90charm; $(LNCMD); $(MAKE) doc )
	( cd fem; $(LNCMD); $(MAKE) doc )
	( cd ampi; $(LNCMD); $(MAKE) doc )
	( cd bluegene; $(LNCMD); $(MAKE) doc )
	( cd mblock; $(LNCMD); $(MAKE) doc )

web:
	( cd install; $(LNCMD); $(MAKE) web )
	( cd converse; $(LNCMD); $(MAKE) web )
	( cd convext; $(LNCMD); $(MAKE) web )
	( cd charm++; $(LNCMD); $(MAKE) web )
	( cd ck; $(LNCMD); $(MAKE) web )
	( cd libraries; $(LNCMD); $(MAKE) web )
	( cd f90charm; $(LNCMD); $(MAKE) web )
	( cd fem; $(LNCMD); $(MAKE) web )
	( cd ampi; $(LNCMD); $(MAKE) web )
	( cd bluegene; $(LNCMD); $(MAKE) web )
	( cd mblock; $(LNCMD); $(MAKE) web )

clean:
	( cd install; $(RMCMD); $(MAKE) clean )
	( cd converse; $(RMCMD); $(MAKE) clean )
	( cd convext; $(RMCMD); $(MAKE) clean )
	( cd charm++; $(RMCMD); $(MAKE) clean )
	( cd ck; $(RMCMD); $(MAKE) clean )
	( cd libraries; $(RMCMD); $(MAKE) clean )
	( cd f90charm; $(RMCMD); $(MAKE) clean )
	( cd fem; $(RMCMD); $(MAKE) clean )
	( cd ampi; $(RMCMD); $(MAKE) clean )
	( cd bluegene; $(RMCMD); $(MAKE) clean )
	( cd mblock; $(RMCMD); $(MAKE) clean )
