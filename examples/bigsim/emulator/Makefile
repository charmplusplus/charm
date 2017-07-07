CHARMRUN=./charmrun
-include ../../common.mk
CHARMC=../../../bin/charmc -language bigsim $(OPTS)
CXX=$(CHARMC)

BINS=jacobi3D maxReduce prime ping octo line ring
BUT_ONE=maxReduce prime ping octo line ring

all: $(BINS)
	$(MAKE) -C littleMD

jacobi3D: common.o
	$(CHARMC) -o $@ $^
$(BUT_ONE): common.o
	$(CHARMC) -no-charmrun -o $@ $^

jacobi3D: jacobi3D.C
maxReduce: maxReduce.C
prime: prime.C
ping: ping.C
octo: octo.C
line: line.C
ring: ring.C

test-local: all
	$(call run, ./maxReduce +p4 +cth3 +wth10 +bgstacksize 102400)
	$(call run, ./octo +p4 3 3 3 4 5  +bgstacksize 102400)
	$(call run, ./line +p4 6 6 6 3 4 10 +bgstacksize 102400)
	$(call run, ./jacobi3D +p4 3 3 3 2 10 0.1 +bgstacksize 204800)
	$(call run, ./prime +p4 3 3 3 3 4 1000 +bgstacksize 102400)
	$(call run, ./ring +p4 2 2 2 1 1)
	$(call run, ./littleMD/bgMD +p4 6 6 6 3 4 +bgstacksize 102400)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f $(BINS)
	rm -f $(patsubst %,%.exe,$(BINS))
	rm -f conv-host charmrun
	$(MAKE) -C littleMD clean
