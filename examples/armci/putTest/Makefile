-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: put

put: put.c
	$(CHARMC) -o put put.c -language armci -balancer RotateLB 

test: put
	$(call run, +p2 ./put +vp2 )

bgtest: put
	$(call run, +p1 ./put +vp2 +x2 +y1 +z1 )

clean:
	rm -f put charmrun *.o charmrun.exe put.exe put.pdb put.ilk
