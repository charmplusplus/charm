all: xi-grammar.tab.h xi-grammar.tab.C xi-scan.C sdag/trans.c

xi-grammar.tab.h xi-grammar.tab.C: xi-grammar.y
	bison -y -d xi-grammar.y
	mv y.tab.c xi-grammar.tab.C
	mv y.tab.h xi-grammar.tab.h

xi-scan.C: xi-scan.l
	flex xi-scan.l
	mv lex.yy.c xi-scan.C

sdag/trans.c: sdag/trans.l
	flex -Psl -osdag/trans.c sdag/trans.l
