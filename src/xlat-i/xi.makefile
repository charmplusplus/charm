all: xi-grammar.tab.h xi-grammar.tab.C xi-scan.C

xi-grammar.tab.h xi-grammar.tab.C: xi-grammar.y
	bison -d xi-grammar.y
	mv xi-grammar.tab.c xi-grammar.tab.C

xi-scan.C: xi-scan.l
	flex xi-scan.l
	mv lex.yy.c xi-scan.C
