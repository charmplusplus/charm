all: xi-grammar.tab.h xi-grammar.tab.C xi-scan.C

xi-grammar.tab.h xi-grammar.tab.C: xi-grammar.y
	yacc -d xi-grammar.y
	mv y.tab.c xi-grammar.tab.C
	mv y.tab.h xi-grammar.tab.h

xi-scan.C: xi-scan.l
	flex xi-scan.l
	mv lex.yy.c xi-scan.C
