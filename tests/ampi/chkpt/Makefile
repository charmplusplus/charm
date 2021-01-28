-include ../../common.mk
CHARMC=../../../bin/ampicc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c $(OPTS)

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

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun ampirun
	rm -rf log
