-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingall multiping

pingall: pingall.C
	$(CHARMC) -o pingall pingall.C -language converse++

multiping: multiping.C
	$(CHARMC) -o multiping multiping.C -language converse++ -no-charmrun

test: all
	$(call run, +p2 ./pingall 0)
	$(call run, +p2 ./pingall 1)
	$(call run, +p2 ./multiping 0 8)
	$(call run, +p2 ./multiping 0 32)
	$(call run, +p2 ./multiping 1 8)
	$(call run, +p2 ./multiping 1 32)

clean:
	rm -f core *.cpm.h *~
	rm -f pingall multiping
	rm -f conv-host charmrun
