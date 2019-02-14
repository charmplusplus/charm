#ifndef HRCTIMER_H
#define HRCTIMER_H

#ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

double inithrc(void);
double gethrctime(void);
uint64_t gethrctime_micro(void);

#ifdef __cplusplus
}
#endif

#endif
