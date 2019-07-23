-include ../../../common.mk
SRC=../../../../src
CHARMC=../../../../bin/charmc $(OPTS)

NAME = largedata

OBJS = $(NAME).o

all: pgm

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS)

pgm.prj: $(OBJS)
	$(CHARMC) -tracemode projections -language charm++ -o pgm.prj $(OBJS)

cifiles: $(NAME).ci
	$(CHARMC) $(NAME).ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h conv-host *.o pgm charmrun cifiles pgm.exe pgm.pdb pgm.ilk

$(NAME).o: $(NAME).C cifiles
	$(CHARMC) -I$(SRC)/conv-core $<

test: all
	$(call run, ./pgm +p4 200000 10)
	$(call run, ./pgm +p6 200000 10)
	$(call run, ./pgm +p4 200000 10 +noCMAForZC)
	$(call run, ./pgm +p6 200000 10 +noCMAForZC)

test-bench: all
	$(call run, ./pgm +p10)

bgtest: all
	$(call run, ./pgm +p2 +x2 +y1 +z1 )
