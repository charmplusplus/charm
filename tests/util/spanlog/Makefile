# Host test of the LB entry-span log (hapi_impl.cpp: CuptiEntryLane prune/compaction, the drains' incremental copies,
# hapiCuptiSpanAt). The functions are extracted from hapi_impl.cpp at build time, so the test follows the source.
# `make test` runs it without and with a span left open for the whole run.
HAPI := ../../../src/arch/cuda/hybridAPI/hapi_impl.cpp
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O1 -Wall

all: spanlog_test

spans_extracted.h: $(HAPI) extract.sh
	./extract.sh $(HAPI) > $@

spanlog_test: spanlog_test.cpp spans_extracted.h
	$(CXX) $(CXXFLAGS) -o $@ spanlog_test.cpp

test: spanlog_test
	./spanlog_test 0
	./spanlog_test 1

clean:
	rm -f spanlog_test spans_extracted.h
