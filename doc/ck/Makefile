PSDIR=../../../ps
PDFDIR=../../../pdf
HDIR=../../../html

all: ps pdf html

ps:
	make clean
	touch index.tex
	latex manual.tex
	latex manual.tex
	if [ -f manual.idx ] ; then makeindex -o index.tex manual.idx ; fi
	latex manual.tex
	dvips -o manual.ps manual.dvi
	/bin/cp manual.ps $(PSDIR)/ck.ps

html:
	make clean
	touch index.tex
	latex manual.tex
	latex2html -local_icons manual.tex
	/bin/rm -rf $(HDIR)/ck
	/bin/cp -R manual $(HDIR)/ck

pdf:
	make clean
	touch index.tex
	pdflatex manual.tex
	if [ -f manual.idx ] ; then makeindex -o index.tex manual.idx ; fi
	pdflatex manual.tex
	/bin/cp manual.pdf $(PDFDIR)/ck.pdf

clean:
	/bin/rm -f *.ps *.pdf *.ilg *.aux *.log *.dvi *.idx *.toc 
	/bin/rm -f index.tex *.blg *.bbl
	/bin/rm -rf manual
