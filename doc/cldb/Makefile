
TEXFILES=HowToLB.tex

ps:
	latex HowToLB.tex
	dvips -o HowToLB.ps HowToLB.dvi

html:
	latex HowToLB.tex
	latex2html HowToLB.tex

install:
	rm -f /expand1/groupMosaic/distrib/cldb.manual.ps.gz
	cp HowToLB.ps /expand1/groupMosaic/distrib/cldb.manual.ps
	gzip /expand1/groupMosaic/distrib/cldb.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/cldb.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/cldb.manual/*
	cp HowToLB/* /expand1/groupMosaic/html-manuals/cldb.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/cldb.manual/*

all: ps html install

co:
	co Makefile
	co $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc

