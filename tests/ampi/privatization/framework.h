#ifndef FRAMEWORK_H_
#define FRAMEWORK_H_

#include "charm-api.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#define privatization_method_str STRINGIZE_VALUE_OF(privatization_method)

#define test_format "#%02d"

CLINKAGE void print_test(int & test, int & rank, const char * name);

#define print_test_fortran FTN_NAME(PRINT_TEST_FORTRAN, print_test_fortran)
FLINKAGE void print_test_fortran(int & test, int & rank, const char * name, long int name_len);

#define test_privatization FTN_NAME(TEST_PRIVATIZATION, test_privatization)
FLINKAGE void test_privatization(int & failed, int & test, int & rank, int & my_wth, int & operation, int & global);
#define test_skip FTN_NAME(TEST_SKIP, test_skip)
FLINKAGE void test_skip(int & test, int & rank);
#define privatization_test_framework FTN_NAME(PRIVATIZATION_TEST_FRAMEWORK, privatization_test_framework)
FLINKAGE void privatization_test_framework(void);

#define perform_test_batch FTN_NAME(PERFORM_TEST_BATCH, perform_test_batch)
FLINKAGE void perform_test_batch(int & failed, int & test, int & rank, int & my_wth, int & operation);

#endif
