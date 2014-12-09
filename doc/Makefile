IDIR	= ../doc
LNCMD	= test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD	= rm -f ./pplmanual.sty
WEBDIR	= /www/manuals
DIRS	= converse convext charm++ libraries f90charm pose \
	ampi bigsim projections tcharm debugger faq \
	bignetsim charj
#	charisma ## uncomment once the manual is updated
#	fem ifem netfem mblock ## dont build regularly. dormant manuals

all: pdf ps html

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

web: doc
	-rm -rf $(WEBDIR)/backup
	mkdir -p $(WEBDIR)/backup && \
	mv $(WEBDIR)/pdf $(WEBDIR)/html $(WEBDIR)/backup
	cp -r pdf html $(WEBDIR) && \
	chgrp -R kale $(WEBDIR)/pdf $(WEBDIR)/html && \
	chmod -R g+w  $(WEBDIR)/pdf $(WEBDIR)/html || \
	{ rm -rf $(WEBDIR)/pdf $(WEBDIR)/html && \
	mv $(WEBDIR)/backup/pdf $(WEBDIR)/backup/html $(WEBDIR); }
	-rm -rf $(WEBDIR)/backup

clean:
	for d in $(DIRS); do (cd $$d; $(RMCMD); $(MAKE) clean) || exit 1; done
	rm -fr ps pdf html
