OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
CHARMC_FLAGS = -D__HIP_PLATFORM_AMD__=ON

LD_LIBS = -module EveryLB

# No .cu here on purpose: these benchmarks only need device allocations and
# host-to-device fills, both of which hapi.h exposes to a plain C++ translation
# unit. Nothing is computed on the GPU, so there is no kernel to compile.
TARGETS = osu_latency_gpu osu_bw_gpu

all: $(TARGETS)

osu_latency_gpu: osu_latency_gpu.o
	$(CHARMC) $(CHARMC_FLAGS) -language charm++ -o $@ $< $(LD_LIBS)

osu_bw_gpu: osu_bw_gpu.o
	$(CHARMC) $(CHARMC_FLAGS) -language charm++ -o $@ $< $(LD_LIBS)

%.decl.h: %.ci
	$(CHARMC) $(CHARMC_FLAGS) $<

osu_latency_gpu.o: osu_latency_gpu.C osu_latency_gpu.decl.h
	$(CHARMC) $(CHARMC_FLAGS) -c $<

osu_bw_gpu.o: osu_bw_gpu.C osu_bw_gpu.decl.h
	$(CHARMC) $(CHARMC_FLAGS) -c $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGETS) charmrun

# Two processes on one host: the placement the benchmark requires. It refuses to
# run in any other layout rather than reporting a number for the wrong path.
test: all
	$(call run, ./osu_latency_gpu +p2 ++ppn 1 +gpushm)
	$(call run, ./osu_bw_gpu +p2 ++ppn 1 +gpushm)
