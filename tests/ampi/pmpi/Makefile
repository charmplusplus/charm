-include ../../common.mk
CHARMC=../../../bin/ampicc $(OPTS)

all: ampi

ampi: pmpi

pmpi: pmpi.c
	$(CHARMC) -o $@ $< $(OPTS)

test: pmpi
	$(call run, +p2 ./pmpi )

clean:
	rm -f *.o pmpi
