TEXFILES= groups.tex ack.tex nodegroups.tex sdag.tex arrays.tex idl.tex seqobjs.tex inhertmplt.tex othercalls.tex intro.tex syntaxchange.tex chares.tex keywords.tex manual.tex messages.tex quiesce.tex further.tex readonly.tex modules.tex overview.tex comprundebug.tex

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
	rm -f /expand1/groupMosaic/distrib/charm++5.0v2.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/charm++5.0v2.ps
	gzip /expand1/groupMosaic/distrib/charm++5.0v2.ps
	chmod 664 /expand1/groupMosaic/distrib/charm++5.0v2.ps.gz
	rm -f /expand1/groupMosaic/distrib/charm++5.0v2.pdf
	cp manual.pdf /expand1/groupMosaic/distrib/charm++5.0v2.pdf
	chmod 664 /expand1/groupMosaic/distrib/charm++5.0v2.pdf
	rm -f /expand1/groupMosaic/html-manuals/charm++5.0v2/*
	cp manual/* /expand1/groupMosaic/html-manuals/charm++5.0v2/
	chmod 664 /expand1/groupMosaic/html-manuals/charm++5.0v2/*

all: ps pdf html install

clean:
	rm -f *.ilg *.aux *.log *.dvi *.idx *.toc index.tex *.blg *.bbl

veryclean: clean
	rm -rf *.ps *.pdf manual/

