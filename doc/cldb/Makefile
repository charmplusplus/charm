ps:
	make clean
	latex HowToLB.tex
	dvips -o HowToLB.ps HowToLB.dvi

html:
	make clean
	latex HowToLB.tex
	latex2html -local_icons HowToLB.tex

pdf:
	make clean
	pdflatex HowToLB.tex

install:
	rm -f /expand1/groupMosaic/distrib/cldb.manual.ps.gz
	cp HowToLB.ps /expand1/groupMosaic/distrib/cldb.manual.ps
	gzip /expand1/groupMosaic/distrib/cldb.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/cldb.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/cldb.manual.pdf
	cp HowToLB.pdf /expand1/groupMosaic/distrib/cldb.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/cldb.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/cldb.manual/*
	cp HowToLB/* /expand1/groupMosaic/html-manuals/cldb.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/cldb.manual/*

all: ps pdf html install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl

veryclean:
	rm -rf *.ps *.pdf HowToLB/

