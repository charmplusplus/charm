IDIR=../../doc

all: ps pdf html

ps:
	if [ ! -d $(IDIR)/ps ] ; then mkdir $(IDIR)/ps ; fi
	( cd install; make ps )
	( cd converse; make ps )
	( cd convext; make ps )
	( cd charm++; make ps )
	( cd cldb; make ps )
	( cd ck; make ps )
	( cd libraries; make ps )
	( cd f90charm; make ps )

pdf:
	if [ ! -d $(IDIR)/pdf ] ; then mkdir $(IDIR)/pdf ; fi
	( cd install; make pdf )
	( cd converse; make pdf )
	( cd convext; make pdf )
	( cd charm++; make pdf )
	( cd cldb; make pdf )
	( cd ck; make pdf )
	( cd libraries; make pdf )
	( cd f90charm; make pdf )

html:
	if [ ! -d $(IDIR)/html ] ; then mkdir $(IDIR)/html ; fi
	( cd install; make html )
	( cd converse; make html )
	( cd convext; make html )
	( cd charm++; make html )
	( cd cldb; make html )
	( cd ck; make html )
	( cd libraries; make html )
	( cd f90charm; make html )

clean:
	( cd install; make clean )
	( cd converse; make clean )
	( cd convext; make clean )
	( cd charm++; make clean )
	( cd cldb; make clean )
	( cd ck; make clean )
	( cd libraries; make clean )
	( cd f90charm; make clean )
