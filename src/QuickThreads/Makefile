
CC=gcc -I. -O2

all: qt stp testpgm

qt:
	ln -s md/sparc.h qtmd.h
	$(CC) -pedantic -Wall -c -o qt.o qt.c
	as -o qtmds.o md/sparc.s
	ar q libqt.a qt.o qtmds.o

stp:
	$(CC) -c stp.c
	ar q libstp.a stp.o

testpgm:
	$(CC) -c meas.c
	as -o qtmdb.o md/sparc_b.s
	$(CC) -o run meas.o qtmdb.o libstp.a libqt.a

mostlyclean:
	rm -f qtmd.h libqt.a libstp.a meas.o stp.o qt.o
	rm -f qtmdb.s qtmds.s qtmdc.o qtmdb.o qtmds.o run

clean: mostlyclean
	rm -f Makefile

