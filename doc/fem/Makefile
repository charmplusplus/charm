# Makefile for LaTeX PPL manual
FILE=manual
TEX=$(FILE).tex
DEST=fem

DOCDIR=..
WEBDIR=/expand8/www/ppl_manuals
WEBARGS=-split 4
LATEX2HTML=latex2html $(WEBARGS) -show_section_numbers \
	-white -antialias -local_icons 


all: ps pdf html

$(FILE).aux: $(TEX)
	latex $(FILE).tex


ps: $(FILE).ps

$(FILE).dvi: $(TEX) $(FILE).aux
	latex $(FILE).tex

$(FILE).ps: $(FILE).dvi
	dvips -t letter -f $(FILE).dvi > $(FILE).ps


pdf: $(FILE).pdf

$(FILE).pdf: $(TEX) $(FILE).aux
	pdflatex $(FILE).tex


html: $(FILE)

$(FILE): $(TEX) $(FILE).aux
	-@ln -s ../pplmanual.* .
	-@rm -fr $(FILE)/*.html $(FILE)/*.aux
	$(LATEX2HTML) $(FILE).tex

clean:
	-rm -fr *.aux *.log *.dvi *.ps *.pdf $(FILE)
	-rm -fr *.log *.bbl *.blg *.toc *~ fig/*.bak

doc: all
	if [ ! -d $(DOCDIR) ] ; then mkdir $(DOCDIR) ; fi
	if [ ! -d $(DOCDIR)/ps ] ; then mkdir $(DOCDIR)/ps ; fi
	if [ ! -d $(DOCDIR)/pdf ] ; then mkdir $(DOCDIR)/pdf ; fi
	if [ ! -d $(DOCDIR)/html ] ; then mkdir $(DOCDIR)/html ; fi
	/bin/cp $(FILE).ps $(DOCDIR)/ps/$(DEST).ps
	/bin/cp $(FILE).pdf $(DOCDIR)/pdf/$(DEST).pdf
	/bin/rm -rf $(DOCDIR)/html/$(DEST)
	/bin/cp -R $(FILE) $(DOCDIR)/html/$(DEST)

web: all
	/bin/cp $(FILE).ps $(WEBDIR)/ps/$(DEST).ps
	/bin/cp $(FILE).pdf $(WEBDIR)/pdf/$(DEST).pdf
	/bin/rm -rf $(WEBDIR)/html/$(DEST)
	/bin/cp -R $(FILE) $(WEBDIR)/html/$(DEST)
	find $(WEBDIR)/html/$(DEST) $(WEBDIR)/p*/$(DEST).p* -type f -exec chmod 664 {} \;
	find $(WEBDIR)/html/$(DEST) -type d -exec chmod 775 {} \;
