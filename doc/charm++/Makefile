TEXFILES= groups.tex ack.tex nodegroups.tex sdag.tex arrays.tex idl.tex seqobjs.tex inhertmplt.tex othercalls.tex intro.tex syntaxchange.tex chares.tex keywords.tex manual.tex messages.tex quiesce.tex further.tex readonly.tex modules.tex overview.tex comprundebug.tex

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
	latex2html -local_icons manual.tex

pdf: $(TEXFILES)
	touch index.tex
	pdflatex manual.tex
	bibtex manual
	makeindex -o index.tex manual.idx
	pdflatex manual.tex
	pdflatex manual.tex

install:
	rm -f /expand1/groupMosaic/distrib/newcharm++.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/newcharm++.manual.ps
	gzip /expand1/groupMosaic/distrib/newcharm++.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/newcharm++.manual.ps.gz
	rm -f /expand1/groupMosaic/distrib/newcharm++.manual.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/newcharm++.manual.pdf
	chmod 664 /expand1/groupMosaic/distrib/newcharm++.manual.pdf
	rm -f /expand1/groupMosaic/html-manuals/newcharm++.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/newcharm++.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/newcharm++.manual/*

all: ps pdf html install

clean:	
	rm -f *.ilg *.aux *.log *.dvi *.ps *.idx *.toc *.pdf index.tex *.blg *.bbl
	rm -rf manual/

