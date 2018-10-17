#ifndef HRCTIMER_H
#define HRCTIMER_H
#include <inttypes.h>
#ifdef __cplusplus
extern "C" {
#endif
double inithrc();
double gethrctime();
uint64_t gethrctime_micro();
#ifdef __cplusplus
}
#endif
#endif
