
all: mostlyclean
	@if [ ! -f Makescript ] ;then echo "Must configure first." ; exit 1; fi
	./Makescript

mostlyclean:
	rm -f qtmd.h libqt.a libstp.a meas.o stp.o qt.o
	rm -f qtmdb.s qtmds.s qtmdc.o qtmdb.o qtmds.o run

clean: mostlyclean
	rm -f Makescript

