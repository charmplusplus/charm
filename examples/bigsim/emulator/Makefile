CHARMC=../../../bin/charmc $(OPTS)

all: jacobi3D maxReduce prime ping octo line ring
	cd littleMD; $(MAKE)

jacobi3D: jacobi3D.C
	$(CHARMC) -language bigsim -o jacobi3D jacobi3D.C 

maxReduce: maxReduce.C
	$(CHARMC) -language bigsim -o maxReduce maxReduce.C

prime: prime.C
	$(CHARMC) -language bigsim -o prime prime.C

ping: ping.C
	$(CHARMC) -language bigsim -o ping ping.C

octo: octo.C
	$(CHARMC) -language bigsim -o octo octo.C

line: line.C
	$(CHARMC) -language bigsim -o line line.C

ring: ring.C
	$(CHARMC) -language bigsim -o ring ring.C

test: all
	./charmrun ./maxReduce +p4 +cth3 +wth10 +bgstacksize 102400 $(TESTOPTS)
	./charmrun +p4 ./octo 3 3 3 4 5  +bgstacksize 102400 $(TESTOPTS)
	./charmrun +p4 ./line 6 6 6 3 4 10 +bgstacksize 102400 $(TESTOPTS)
	./charmrun +p4 ./jacobi3D 3 3 3 2 10 0.1 +bgstacksize 204800 $(TESTOPTS)
	./charmrun +p4 ./prime 3 3 3 3 4 1000 +bgstacksize 102400 $(TESTOPTS)
	./charmrun +p4 ./ring 2 2 2 1 1 $(TESTOPTS)
	./charmrun +p4 littleMD/bgMD 6 6 6 3 4 +bgstacksize 102400 $(TESTOPTS)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f jacobi3D maxReduce prime ping octo line ring
	rm -f jacobi3D.exe maxReduce.exe prime.exe ping.exe octo.exe line.exe ring.exe
	rm -f conv-host charmrun
	cd littleMD; $(MAKE) clean
