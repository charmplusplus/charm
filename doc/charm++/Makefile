DOCDIR=../../../doc
WEBDIR=/expand6/groupMosaic/ppl_manuals

all: ps pdf html

ps:
	touch index.tex
	latex manual.tex
	latex manual.tex
	if [ -f manual.idx ] ; then makeindex -o index.tex manual.idx ; fi
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	touch index.tex
	-@ln -s ../pplmanual.* .
	latex manual.tex
	latex2html -local_icons manual.tex

pdf:
	touch index.tex
	pdflatex manual.tex
	if [ -f manual.idx ] ; then makeindex -o index.tex manual.idx ; fi
	pdflatex manual.tex

doc:
	make all
	if [ ! -d $(DOCDIR) ] ; then mkdir $(DOCDIR) ; fi
	if [ ! -d $(DOCDIR)/ps ] ; then mkdir $(DOCDIR)/ps ; fi
	if [ ! -d $(DOCDIR)/pdf ] ; then mkdir $(DOCDIR)/pdf ; fi
	if [ ! -d $(DOCDIR)/html ] ; then mkdir $(DOCDIR)/html ; fi
	/bin/cp manual.ps $(DOCDIR)/ps/charm++.ps
	/bin/cp manual.pdf $(DOCDIR)/pdf/charm++.pdf
	/bin/rm -rf $(DOCDIR)/html/charm++
	/bin/cp -R manual $(DOCDIR)/html/charm++

web:
	make all
	/bin/cp manual.ps $(WEBDIR)/ps/charm++.ps
	/bin/cp manual.pdf $(WEBDIR)/pdf/charm++.pdf
	/bin/rm -rf $(WEBDIR)/html/charm++
	/bin/cp -R manual $(WEBDIR)/html/charm++
	find $(WEBDIR) -type f -exec chmod 664 {} \;
	find $(WEBDIR) -type d -exec chmod 775 {} \;

clean:
	/bin/rm -f *.ps *.pdf *.ilg *.aux *.log *.dvi *.idx *.toc 
	/bin/rm -f index.tex *.blg *.bbl
	/bin/rm -rf manual
