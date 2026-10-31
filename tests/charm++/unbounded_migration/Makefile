-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: migrate

migrate: migrate.decl.h migrate.C
	$(CHARMC) migrate.C -o migrate 

migrate.decl.h: migrate.ci
	$(CHARMC) migrate.ci

test: migrate
	$(call run, +p4 ./migrate)

smptest: migrate
	$(call run, +p4 ./migrate)

clean:
	rm -rf *.decl.h *.def.h migrate charmrun
