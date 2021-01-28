all: xi-grammar.tab.h xi-grammar.tab.C xi-scan.C 
#sdag/sdag-trans.c

xi-grammar.tab.h xi-grammar.tab.C: xi-grammar.y
	bison -y -d xi-grammar.y
	cp y.tab.c xi-grammar.tab.C
	cp y.tab.h xi-grammar.tab.h
	rm y.tab.c y.tab.h

xi-scan.C: xi-scan.l
	flex xi-scan.l
	cp lex.yy.c xi-scan.C
	rm lex.yy.c

sdag/sdag-trans.c: sdag/trans.l
	flex -Psl -osdag/sdag-trans.c sdag/trans.l

clean:
	rm -rf xi-grammar.tab.h xi-grammar.tab.C xi-scan.C
