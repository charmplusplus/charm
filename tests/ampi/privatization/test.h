#ifndef TEST_H_
#define TEST_H_

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#define privatization_method_str STRINGIZE_VALUE_OF(privatization_method)

#define result_indent "  "

extern "C" void test_privatization_(int & failed, int & rank, int & my_wth, int & global);
extern "C" void privatization_test_framework_(void);

extern "C" void perform_test_batch_(int & failed, int & rank, int & my_wth);

#endif
