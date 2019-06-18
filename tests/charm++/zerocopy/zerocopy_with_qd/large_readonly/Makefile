-include ../../../../common.mk
CHARMDIR = ../../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: ro_qd_test

OBJS = ro_qd_test.o

ro_qd_test: $(OBJS)
	$(CHARMC) -language charm++ -o ro_qd_test $(OBJS)

cifiles: ro_qd_test.ci
	$(CHARMC)  ro_qd_test.ci
	touch cifiles

ro_qd_test.o: ro_qd_test.C cifiles
	$(CHARMC) -c ro_qd_test.C

test: all
	$(call run, +p1 ./ro_qd_test 60)
	$(call run, +p4 ./ro_qd_test 100)
	$(call run, +p6 ./ro_qd_test 100)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f ro_qd_test charmrun cifiles
