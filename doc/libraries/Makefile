TEXFILES=manual.tex intro.tex irecv.tex

ps: $(TEXFILES)
	make clean
	touch index.tex
	latex manual.tex
	bibtex manual
	makeindex -o index.tex manual.idx
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html: $(TEXFILES)
	make clean
	latex manual.tex
	latex2html -local_icons manual.tex

pdf: $(TEXFILES)
	make clean
	touch index.tex
	pdflatex manual.tex
	bibtex manual
	makeindex -o index.tex manual.idx
	pdflatex manual.tex
	pdflatex manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/libraries.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/libraries.manual.ps
	gzip /expand1/groupMosaic/distrib/libraries.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/libraries.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/libraries.manual.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/libraries.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/libraries.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/libraries.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/libraries.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/libraries.manual/*

all: ps pdf html install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl

veryclean: clean
	rm -rf *.ps *.pdf manual/

