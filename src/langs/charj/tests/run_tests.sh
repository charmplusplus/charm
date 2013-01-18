#!/bin/bash

tests=0
passed_tests=0

BASE_DIR=`pwd`
CHARJC="$BASE_DIR/../bin/charjc"

function test_file() {
    tests=`expr $tests + 1`
    file="$1"
    EXPECT="success"
    head -n 1 $file | grep -v "fail" > /dev/null
    if [ "$?" -ne "0" ]; then
        EXPECT="failure"
    fi

    echo "Test $tests: $file" >> $BASE_DIR/run_tests.log
    RESULT="success"
    $CHARJC $file &> run_tests.log.tmp
    if [ "$?" -ne "0" ]; then
        RESULT="failure"
    fi
	count=`cat run_tests.log.tmp | wc -l`
	if [ $count -gt 2 ]; then
		RESULT="failure"
	fi
    cat run_tests.log.tmp >> $BASE_DIR/run_tests.log
    rm run_tests.log.tmp

    PASS="FAIL"
    if [ "$RESULT" = "$EXPECT" ]; then
        PASS="OK"
        passed_tests=`expr $passed_tests + 1`
    fi
    echo "$PASS" >> run_tests.log
    printf "Testing %s...%s\n" `basename $file` $PASS
}

rm -f run_tests.log
TESTDIRS="functional"
for dir in $TESTDIRS; do
    pushd $dir > /dev/null
    rm -rf *.gen *.decl.h
    for file in `find . -name "*.cj"`; do
        test_file "$file"
    done
    popd > /dev/null
done
echo "Passed $passed_tests/$tests tests"

