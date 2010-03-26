IDIR	= ../doc
LNCMD	= test ! -f pplmanual.sty && ln -f -s ../pplmanual.sty .
RMCMD	= rm -f ./pplmanual.sty
WEBDIR	= /www/manuals
DIRS	= install converse convext charm++ libraries f90charm charisma pose \
	fem ifem netfem ampi bigsim mblock projections tcharm debugger faq \
	bignetsim

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
	-rm -rf $(WEBDIR)/ps.new $(WEBDIR)/pdf.new $(WEBDIR)/html.new
	-rm -rf $(WEBDIR)/ps.old $(WEBDIR)/pdf.old $(WEBDIR)/html.old
	-rm -rf $(WEBDIR)/ps.old.nobackup $(WEBDIR)/pdf.old.nobackup $(WEBDIR)/html.old.nobackup
	cp -r ps $(WEBDIR)/ps.new
	cp -r pdf $(WEBDIR)/pdf.new
	cp -r html $(WEBDIR)/html.new
	cp manual.css $(WEBDIR)/
	chgrp -R kale $(WEBDIR)/ps.new $(WEBDIR)/pdf.new $(WEBDIR)/html.new
	-chgrp -R kale $(WEBDIR)/manual.css
	chmod -R g+w $(WEBDIR)/ps.new $(WEBDIR)/pdf.new $(WEBDIR)/html.new
	-chmod -R g+w $(WEBDIR)/manual.css
	mv $(WEBDIR)/ps $(WEBDIR)/ps.old.nobackup
	mv $(WEBDIR)/pdf $(WEBDIR)/pdf.old.nobackup
	mv $(WEBDIR)/html $(WEBDIR)/html.old.nobackup
	mv $(WEBDIR)/ps.new $(WEBDIR)/ps
	mv $(WEBDIR)/pdf.new $(WEBDIR)/pdf
	mv $(WEBDIR)/html.new $(WEBDIR)/html

clean:
	for d in $(DIRS); do (cd $$d; $(LNCMD); $(MAKE) clean) || exit 1; done
	rm -fr ps pdf html
