# Build the tests using parallel make
add_test(build-tests      "make" "-C" "../tests"      "-j4"  "OPTS=\"-g\"")
add_test(build-examples   "make" "-C" "../examples"   "-j4"  "OPTS=\"-g\"")
add_test(build-benchmarks "make" "-C" "../benchmarks" "-j4"  "OPTS=\"-g\"")

# Run the tests
add_test(tests/ "make" "-C" "../tests" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(tests/ PROPERTIES  TIMEOUT "2400")
add_test(examples/ "make" "-C" "../examples" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(examples/ PROPERTIES  TIMEOUT "2400")
add_test(benchmarks/ "make" "-C" "../benchmarks" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(benchmarks/ PROPERTIES  TIMEOUT "2400")
