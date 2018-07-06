#ifndef TEST_H_
#define TEST_H_

#include "charm-api.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#define privatization_method_str STRINGIZE_VALUE_OF(privatization_method)

#define result_indent "  "

#define test_privatization FTN_NAME(TEST_PRIVATIZATION, test_privatization)
FDECL void test_privatization(int & failed, int & rank, int & my_wth, int & global);
#define privatization_test_framework FTN_NAME(PRIVATIZATION_TEST_FRAMEWORK, privatization_test_framework)
FDECL void privatization_test_framework(void);

#define perform_test_batch FTN_NAME(PERFORM_TEST_BATCH, perform_test_batch)
FDECL void perform_test_batch(int & failed, int & rank, int & my_wth);

#endif
