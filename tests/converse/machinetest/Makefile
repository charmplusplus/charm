CHARMC=../../../bin/charmc $(OPTS)

all: pingall multiping

pingall: pingall.C
	$(CHARMC) -o pingall pingall.C -language converse++

multiping: multiping.C
	$(CHARMC) -o multiping multiping.C -language converse++

test: all
	./charmrun +p2 ./pingall 0
	./charmrun +p2 ./pingall 1
	./charmrun +p2 ./multiping 0 16
	./charmrun +p2 ./multiping 0 128
	./charmrun +p2 ./multiping 1 16
	./charmrun +p2 ./multiping 1 128

clean:
	rm -f core *.cpm.h *~
	rm -f pingall multiping
	rm -f conv-host charmrun
