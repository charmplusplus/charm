all: xi-parse.tab.C xi-scan.C

xi-parse.tab.C: xi-parse.bison
	byacc -d xi-parse.bison
	mv y.tab.c xi-parse.tab.C
	mv y.tab.h xi-parse.tab.h

xi-scan.C: xi-scan.flex
	flex xi-scan.flex
	mv lex.yy.c xi-scan.C
