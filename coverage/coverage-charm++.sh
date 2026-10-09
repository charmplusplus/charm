#!/bin/bash

# it is possible to run the coverage across multiple builds, but the
# coverage process is designed to report that SomeBinary has coverage
# percentage Y.  Not, source file X has sum(across all compilations
# and tests) coverage.  So what you end up with is a discrete coverage
# calculation for each platform for each source file.  Which makes it
# pretty pointless to run this on more than one platform at a time.

# However, if you want to build a bunch at one, here is a representative set
declare -a BUILDS=("multicore-linux-x86_64" "netlrts-linux-x86_64" "netlrts-linux-x86_64-smp")

if [[ "$#" -gt 0 ]]; then
    declare -a BUILDS
    while [[ "$#" > 0 ]]
    do
        BUILDS+=("$1");
        shift;
        echo "BUILD now $BUILDS";
        echo "ARGC now $#";
    done
else
    declare -a BUILDS=("netlrts-linux-x86_64-smp")
fi

echo "testing builds $BUILDS"

/bin/rm -f summary.txt

covdir="$PWD"

for b in "${BUILDS[@]}" ; do
    covbuild="$b-covbuild"
    echo "coverage base directory $covdir buildir $covbuild"
    /bin/rm -f "$b"-cov.info
    cd ../
    if [ -d "$covbuild" ] ; then
        /bin/rm -rf "$covbuild"
    fi
    ./buildold charm++ "$b" -j8 --suffix "covbuild" --coverage -O0 -save -fprofile-update=atomic
    cd "$covbuild/tests/charm++"
    make -j8 OPTS+="--coverage -save" TESTOPTS+="++local" test
    cd "../../examples/charm++"
    make -j8 OPTS+="--coverage -save" TESTOPTS+="++local" test
    cd ../../tmp/
#    make -j8 test TESTOPTS+="++local"
    testname="$(cut -d'-' -f1 <<<"$b")"
    testsuff="$(cut -d'-' -f4 <<<"$b")"
    if [ "$testsuff" = "smp" ] ; then
       testname=$testsuff
    fi
    echo "test $testname"
    cd ../..
    lcov --capture --directory "$covbuild"/tmp --rc branch_coverage=1 --output-file "$covdir"/"$b"-cov.info --test-name "$testname" --rc geninfo_unexecuted_blocks=1 --ignore-errors mismatch --ignore-errors negative
    # lcov --add-tracefile "$b"-cov.info --output-file coverage/"$b"-total.info >> summary.txt
    cd "$covdir"
    genhtml "$covdir/$b-cov.info" --description-file tests.desc -o "$covdir/$b"-cov-html >> gensummary.txt
done


/bin/rm -f ckconv

gcc ck-convcore-summary.C -o ckconv
for b in "${BUILDS[@]}" ; do
    ./ckconv "$b"-cov.info >> summary.txt
done
