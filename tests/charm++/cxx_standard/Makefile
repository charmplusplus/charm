CHARMC=../../../bin/charmc $(OPTS)

all test smptest testp: cxx_standard

cxx_standard: cxx_standard.C
	$(CHARMC) -language converse++ -o cxx_standard cxx_standard.C

clean:
	rm -f cxx_standard
