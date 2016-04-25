-include ../../common.mk
CHARMC=../../../bin/charmc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language ampi $(OPTS)

test: hello
	rm -rf log
	$(call run, +p4 ./hello +vp8 )
	-sync
	$(call run, +p4 ./hello +vp8 +restart log )
	$(call run, +p2 ./hello +vp8 +restart log )
	-sync
	-rm -fr log
	$(call run, +p2 ./hello +vp8 )
	-sync
	$(call run, +p4 ./hello +vp8 +restart log )

bgtest: clean ampi
	$(call run, +p4 ./hello +vp8 +x4 +y1 +z1)
	$(call run, +p4 ./hello +vp8 +restart log +x4 +y1 +z1)
	$(call run, +p2 ./hello +vp8 +restart log +x2 +y1 +z1)

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun
	rm -rf log
