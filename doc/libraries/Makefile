TEXFILES=manual.tex intro.tex

ps: $(TEXFILES)
	touch index.tex
	latex manual.tex
	bibtex manual
	makeindex -o index.tex manual.idx
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html: $(TEXFILES)
	latex manual.tex
	latex2html manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/libraries.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/libraries.manual.ps
	gzip /expand1/groupMosaic/distrib/libraries.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/libraries.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/libraries.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/libraries.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/libraries.manual/*

all: ps html install

co:
	co Makefile
	co $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc

