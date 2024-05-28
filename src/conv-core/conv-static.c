/* Things that must be statically linked to the binary, even in shared mode. */

#include "conv-config.h"
#include <stddef.h>

/* __executable_start has symbol visibility type STV_HIDDEN */
/* https://github.com/charmplusplus/charm/issues/1893 */
extern void* CmiExecutableStart;
#if CMK_HAS_EXECUTABLE_START
extern char __executable_start;
void* CmiExecutableStart = (void*)&__executable_start;
#else
void* CmiExecutableStart = NULL;
#endif
