-include ../../common.mk
OPTS=-O3 -g
CHARMBASE=../../../
CHARMC=$(CHARMBASE)/bin/charmc -language ampi $(OPTS)
MPICC=mpicc $(OPTS)

all: intercomm_coll

intercomm_coll: intercomm_coll.c
	$(CHARMC) intercomm_coll.c -language ampi  -o intercomm_coll

test: all
	$(call run, +p1 ./intercomm_coll +vp4 ++local)

clean:
	rm -rf charmrun conv-host moduleinit* *.o intercomm_coll *~ *.sts core

