TEXFILES= groups.tex related.tex ack.tex nodegroups.tex sdag.tex arrays.tex idl.tex seqobjs.tex inhertmplt.tex othercalls.tex intro.tex syntaxchange.tex chares.tex keywords.tex tools.tex contacts.tex manual.tex messages.tex quiesce.tex further.tex readonly.tex modules.tex overview.tex comprundebug.tex

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
	rm -f /expand1/groupMosaic/distrib/newcharm++.manual.ps.gz
	cp manual.ps /expand1/groupMosaic/distrib/newcharm++.manual.ps
	gzip /expand1/groupMosaic/distrib/newcharm++.manual.ps
	chmod 664 /expand1/groupMosaic/distrib/newcharm++.manual.ps.gz
	rm -f /expand1/groupMosaic/html-manuals/newcharm++.manual/*
	cp manual/* /expand1/groupMosaic/html-manuals/newcharm++.manual/
	chmod 664 /expand1/groupMosaic/html-manuals/newcharm++.manual/*

all: ps html install

co:
	co Makefile
	co $(TEXFILES)

clean:	
	rm -f *.aux *.log *.dvi *.ps *.idx *.toc

