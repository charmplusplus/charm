-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) -g

all: pgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o  -language ParFUM

pgm.o: pgm.C
	$(CHARMC) -c pgm.C


test: pgm
	$(call run, ./pgm +vp4 +p2)

bgtest: pgm
	$(call run, ./pgm +vp4 +p2 +x2 +y2 +z1)

clean:
	rm -f pgm fpgm *.o conv-host charmrun
