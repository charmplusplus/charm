
TEXFILES=manual.tex ack.tex

ps:
	make clean
	latex manual.tex
	latex manual.tex
	latex manual.tex
	dvips -o manual.ps manual.dvi

html:
	make clean
	latex manual.tex
	latex2html -local_icons manual.tex

pdf:
	make clean
	pdflatex manual.tex
	pdflatex manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/kernel.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/kernel.manual.ps
	gzip /expand1/groupMosaic/distrib/kernel.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/kernel.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/kernel.manual.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/kernel.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/kernel.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/kernel.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/kernel.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/kernel.manual/*

all: ps pdf html install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl

veryclean:
	rm -rf *.ps *.pdf manual/

