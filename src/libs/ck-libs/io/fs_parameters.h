#ifndef FS_PARAMETERS_H
#define FS_PARAMETERS_H
#include <stdlib.h>
#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t CkGetFileStripeSize(const char *filename);

#ifdef __cplusplus
}
#endif

#endif
