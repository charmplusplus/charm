ps:
	latex HowToLB.tex
	dvips -o HowToLB.ps HowToLB.dvi

html:
	latex HowToLB.tex
	latex2html -local_icons HowToLB.tex

pdf:
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

all: ps html install

clean:	
	rm -rf *.aux *.log *.dvi *.ps *.idx *.toc *.pdf HowToLB

