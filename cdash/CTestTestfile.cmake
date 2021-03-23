# Determine where the tests/ examples/ benchmarks/ directories are.
# Necessary for Windows, which does not have the usual tmp etc. symlinks.
file(GLOB_RECURSE mydir FOLLOW_SYMLINKS  ../*/include/charm-version.h) # The '*' is the dir we want.
get_filename_component(mydir ${mydir} DIRECTORY)

# Build the tests using parallel make
add_test(build-tests      "make" "-C" "${mydir}/../tests"      "-j4"  "OPTS=\"-g\"")
add_test(build-examples   "make" "-C" "${mydir}/../examples"   "-j4"  "OPTS=\"-g\"")
add_test(build-benchmarks "make" "-C" "${mydir}/../benchmarks" "-j4"  "OPTS=\"-g\"")

# Run the tests
add_test(tests/ "make" "-C" "${mydir}/../tests" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(tests/ PROPERTIES  TIMEOUT "2400")
add_test(examples/ "make" "-C" "${mydir}/../examples" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(examples/ PROPERTIES  TIMEOUT "2400")
add_test(benchmarks/ "make" "-C" "${mydir}/../benchmarks" "test" "TESTOPTS=$ENV{AUTOBUILD_TEST_OPTS}")
set_tests_properties(benchmarks/ PROPERTIES  TIMEOUT "2400")
