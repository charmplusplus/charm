IDIR=../../doc
LNCMD=test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD=rm -f ./pplmanual.sty
DIRS=install converse convext charm++ libraries f90charm \
	fem ifem ampi bluegene mblock projections tcharm

all: ps pdf html

ps:
	[ -d $(IDIR)/ps ] || mkdir $(IDIR)/ps
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) ps) || exit 1; done

pdf:
	[ -d $(IDIR)/pdf ] || mkdir $(IDIR)/pdf
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) pdf) || exit 1; done

html:
	[ -d $(IDIR)/html ] || mkdir $(IDIR)/html
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) html) || exit 1; done

doc:
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) doc) || exit 1; done

web:
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) web) || exit 1; done

clean:
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) clean) || exit 1; done
	rm -fr ps pdf html
