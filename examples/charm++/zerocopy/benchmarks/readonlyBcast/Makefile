-include ../../../../common.mk
CHARMC=../../../../../bin/charmc $(OPTS)

all:	readonlyBcast readonlyZCBcast

OBJS1 = readonlyBcast.o

OBJS2 = readonlyZCBcast.o

readonlyBcast: $(OBJS1)
	$(CHARMC) -language charm++ -o readonlyBcast $(OBJS1)

readonlyZCBcast: $(OBJS2)
	$(CHARMC) -language charm++ -o readonlyZCBcast $(OBJS2)

cifiles: readonlyBcast.ci
	$(CHARMC)  readonlyBcast.ci
	touch cifiles

readonlyBcast.o: readonlyBcast.C cifiles
	$(CHARMC) -c readonlyBcast.C -DCMK_ONESIDED_RO_DISABLE=1 -o readonlyBcast.o

readonlyZCBcast.o: readonlyBcast.C cifiles
	$(CHARMC) -c readonlyBcast.C -DCMK_ONESIDED_RO_DISABLE=0 -o readonlyZCBcast.o

test: all
	$(call run, +p4 ./readonlyBcast 2000)
	$(call run, +p4 ./readonlyZCBcast 2000)

test-bench: all
	$(call run, +p4 ./readonlyBcast 32)
	$(call run, +p4 ./readonlyZCBcast 32)

	$(call run, +p4 ./readonlyBcast 256)
	$(call run, +p4 ./readonlyZCBcast 256)

	$(call run, +p4 ./readonlyBcast 2048)
	$(call run, +p4 ./readonlyZCBcast 2048)

	$(call run, +p4 ./readonlyBcast 16384)
	$(call run, +p4 ./readonlyZCBcast 16384)

	$(call run, +p4 ./readonlyBcast 131072)
	$(call run, +p4 ./readonlyZCBcast 131072)

	$(call run, +p4 ./readonlyBcast 1048576)
	$(call run, +p4 ./readonlyZCBcast 1048576)

	$(call run, +p4 ./readonlyBcast 8388576)
	$(call run, +p4 ./readonlyZCBcast 8388576)

	$(call run, +p8 ./readonlyBcast 32)
	$(call run, +p8 ./readonlyZCBcast 32)

	$(call run, +p8 ./readonlyBcast 256)
	$(call run, +p8 ./readonlyZCBcast 256)

	$(call run, +p8 ./readonlyBcast 2048)
	$(call run, +p8 ./readonlyZCBcast 2048)

	$(call run, +p8 ./readonlyBcast 16384)
	$(call run, +p8 ./readonlyZCBcast 16384)

	$(call run, +p8 ./readonlyBcast 131072)
	$(call run, +p8 ./readonlyZCBcast 131072)

	$(call run, +p8 ./readonlyBcast 1048576)
	$(call run, +p8 ./readonlyZCBcast 1048576)

	$(call run, +p8 ./readonlyBcast 8388576)
	$(call run, +p8 ./readonlyZCBcast 8388576)

clean:
	rm -f *.decl.h *.def.h conv-host *.o readonlyBcast readonlyZCBcast charmrun cifiles
