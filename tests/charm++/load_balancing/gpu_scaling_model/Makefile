-include ../../../common.mk

CHARMC = ../../../../bin/charmc $(OPTS)

all: gpu_scaling_model

gpu_scaling_model: gpu_scaling_model.o
	$(CHARMC) -language charm++ -o $@ $<

gpu_scaling_model.decl.h gpu_scaling_model.def.h: gpu_scaling_model.ci
	$(CHARMC) gpu_scaling_model.ci

gpu_scaling_model.o: gpu_scaling_model.C gpu_scaling_model.decl.h
	$(CHARMC) -c gpu_scaling_model.C

test: gpu_scaling_model
	$(call run, +p1 ./gpu_scaling_model)

testp: test

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o charmrun charmrun.exe gpu_scaling_model \
	  gpu_scaling_model.exe gpu_scaling_model.pdb gpu_scaling_model.ilk
