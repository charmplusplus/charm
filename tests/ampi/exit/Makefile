-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS) $(MOPTS)

all: test_exit

test_exit: test_exit.C
	$(CHARMC) -o $@ test_exit.C

# Check that application exits correctly with exit code 42.
test: clean test_exit
	$(call run, test_exit 2>&1)| tee tmp.out; grep -q "Application terminated with exit code 42" tmp.out && echo "Exit test passed." || (echo "Exit test failed."; false)
	@rm -f tmp.out

clean:
	rm -f *.o test_exit charmrun ampirun
