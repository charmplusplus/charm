-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)
COMMON_OBJS=	config.o mesh.o node.o \
	lst_NL.o lst_coh2.o
FEM_OBJS=fem_main.o fem_mesh.o $(COMMON_OBJS)
SERIAL_OBJS=serial_main.o $(COMMON_OBJS)

all: pgm

# Common files:
config.o: config.C crack.h
	$(CHARMC) -c config.C

mesh.o: mesh.C crack.h
	$(CHARMC) -c mesh.C

node.o: node.C crack.h
	$(CHARMC) -c node.C

lst_NL.o: lst_NL.C crack.h
	$(CHARMC) -c lst_NL.C

lst_coh2.o: lst_coh2.C crack.h
	$(CHARMC) -c lst_coh2.C


# FEM parallel program
fem_main.o: fem_main.C crack.h
	$(CHARMC) -c fem_main.C

fem_mesh.o: fem_mesh.C crack.h
	$(CHARMC) -c fem_mesh.C

pgm: $(FEM_OBJS)
	$(CHARMC) -o pgm $(FEM_OBJS) -language fem -module netfem

# Simple serial program
serial_main.o: serial_main.C crack.h
	$(CHARMC) -c serial_main.C

serial: $(SERIAL_OBJS)
	$(CHARMC) -o serial $(SERIAL_OBJS) -seq -language c++


test: pgm
	$(call run, ./pgm +p2 +vp4 )

bgtest: pgm
	$(call run, ./pgm +p2 +vp4 +x2 +y2 +z1 )

clean:
	rm -rf *.o pgm *.sts *.log *.bak *~ SunWS_cache ii_files ti_files
	rm -f conv-host getmesh charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

veryclean: clean
	rm -f *.mesh *.graph meshdata.*
