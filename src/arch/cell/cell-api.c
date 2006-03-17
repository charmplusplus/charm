#include "cell-api.h"

#if CMK_CELL
void offloadCallback(void * data)
{
 CthThread tid = (CthThread)data;
 CthAwaken(tid);
} 
#endif
