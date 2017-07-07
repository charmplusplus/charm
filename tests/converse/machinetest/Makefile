CHARMRUN=./charmrun
-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingall multiping

pingall: pingall.C
	$(CHARMC) -o pingall pingall.C -language converse++

multiping: multiping.C
	$(CHARMC) -o multiping multiping.C -language converse++ -no-charmrun

test-local: all
	$(call run, ./pingall +p2 0)
	$(call run, ./pingall +p2 1)
	$(call run, ./multiping +p2 0 8)
	$(call run, ./multiping +p2 0 32)
	$(call run, ./multiping +p2 1 8)
	$(call run, ./multiping +p2 1 32)

clean:
	rm -f core *.cpm.h *~
	rm -f pingall multiping
	rm -f conv-host charmrun
