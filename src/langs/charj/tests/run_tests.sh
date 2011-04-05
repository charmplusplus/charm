#!/bin/sh

tests=0
passed_tests=0

function test_file() {
    tests=`expr $tests + 1`
    file="$1"
    EXPECT="success"
    head -n 1 $file | grep -v "fail" > /dev/null
    if [ "$?" -ne "0" ]; then
        EXPECT="failure"
    fi

    echo "Test $tests: $file" >> run_tests.log
    RESULT="success"
    ../bin/charjc $file &> run_tests.log.tmp
    if [ "$?" -ne "0" ]; then
        RESULT="failure"
    fi
    cat run_tests.log.tmp >> run_tests.log
    rm run_tests.log.tmp

    PASS="FAIL"
    if [ "$RESULT" = "$EXPECT" ]; then
        PASS="OK"
        passed_tests=`expr $passed_tests + 1`
    fi
    echo "$PASS" >> run_tests.log
    printf "Testing %s, expected %s...%s\n" `basename $file` $EXPECT $PASS
}

rm -f run_tests.log
TESTDIRS="unit"
TESTFILES=`find $TESTDIRS -name "*.cj"`
for f in $TESTFILES; do
    test_file "$f"
done
echo "Passed $passed_tests/$tests tests"

