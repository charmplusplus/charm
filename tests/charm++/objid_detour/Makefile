-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: detour

detour: detour.decl.h detour.C
	$(CHARMC) detour.C -o detour

detour.decl.h: detour.ci
	$(CHARMC) detour.ci

# Needs two processes (CI's lcrun -n 2 runs) or more; one process skips.
# +messageBufferingThreshold 0: a message that finds no entry is held and the
# location asked for, instead of being forwarded to the home.
test: detour
	$(call run, ./detour +p4)
	$(call run, ./detour +p4 +messageBufferingThreshold 0)

smptest: detour
	$(call run, ./detour +p4)

clean:
	rm -rf *.decl.h *.def.h detour charmrun
