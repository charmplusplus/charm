PSDIR=../../../doc/ps
PDFDIR=../../../doc/pdf
HDIR=../../../doc/html

all: ps pdf html

ps:
	make clean
	touch index.tex
	latex HowToLB.tex
	latex HowToLB.tex
	if [ -f HowToLB.idx ] ; then makeindex -o index.tex HowToLB.idx ; fi
	latex HowToLB.tex
	dvips -o HowToLB.ps HowToLB.dvi
	/bin/cp HowToLB.ps $(PSDIR)/cldb.ps

html:
	make clean
	touch index.tex
	latex HowToLB.tex
	latex2html -local_icons HowToLB.tex
	/bin/rm -rf $(HDIR)/cldb
	/bin/cp -R HowToLB $(HDIR)/cldb

pdf:
	make clean
	touch index.tex
	pdflatex HowToLB.tex
	if [ -f HowToLB.idx ] ; then makeindex -o index.tex HowToLB.idx ; fi
	pdflatex HowToLB.tex
	/bin/cp HowToLB.pdf $(PDFDIR)/cldb.pdf

clean:
	/bin/rm -f *.ps *.pdf *.ilg *.aux *.log *.dvi *.idx *.toc 
	/bin/rm -f index.tex *.blg *.bbl
	/bin/rm -rf HowToLB
