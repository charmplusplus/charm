CHARMC=../../../bin/charmc $(OPTS)

all: libmp1.a driver

driver: driver.o
	$(CHARMC) driver.o libmp1.a -o driver -language charm++ -std=c++1z

driver.o: driver.C driver.def.h driver.decl.h
	$(CHARMC) -c driver.C -g

driver.def.h driver.decl.h: driver.ci
	$(CHARMC) driver.ci

libmp1.a: htram.o
	$(CHARMC) htram.o -o libmp1.a -language charm++ 

htram.o : htram.C htram.def.h htram.decl.h
	$(CHARMC) -c htram.C -g

htram.def.h htram.decl.h : htram.ci
	$(CHARMC) htram.ci

test: all
	./charmrun +p16 ./driver  ++local +setcpuaffinity ++ppn 4

clean:
	rm -f conv-host *.o charmrun mp1
	rm -f *.def.h *.decl.h
	rm -f *.log.gz *.projrc *.topo *.sts *.sum
	rm libmp1.a

