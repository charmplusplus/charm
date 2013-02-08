CHARMC=../../../bin/charmc -language bigsim $(OPTS)
CXX=$(CHARMC)

BINS=jacobi3D maxReduce prime ping octo line ring
BUT_ONE=maxReduce prime ping octo line ring

all: $(BINS)
	cd littleMD; $(MAKE)

$(BINS): common.o
	$(CHARMC) -o $@ $^

$(BUT_ONE): OPTS+=-no-charmrun
jacobi3D: jacobi3D.C
maxReduce: maxReduce.C
prime: prime.C
ping: ping.C
octo: octo.C
line: line.C
ring: ring.C

RUN=./charmrun +p4

test: all
	$(RUN) ./maxReduce +cth3 +wth10 +bgstacksize 102400 $(TESTOPTS)
	$(RUN) ./octo 3 3 3 4 5  +bgstacksize 102400 $(TESTOPTS)
	$(RUN) ./line 6 6 6 3 4 10 +bgstacksize 102400 $(TESTOPTS)
	$(RUN) ./jacobi3D 3 3 3 2 10 0.1 +bgstacksize 204800 $(TESTOPTS)
	$(RUN) ./prime 3 3 3 3 4 1000 +bgstacksize 102400 $(TESTOPTS)
	$(RUN) ./ring 2 2 2 1 1 $(TESTOPTS)
	$(RUN) littleMD/bgMD 6 6 6 3 4 +bgstacksize 102400 $(TESTOPTS)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f $(BINS)
	rm -f $(patsubst %,%.exe,$(BINS))
	rm -f conv-host charmrun
	cd littleMD; $(MAKE) clean
