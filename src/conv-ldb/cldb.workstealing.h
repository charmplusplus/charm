#include "cldb.h"
/* for sqrt() */
#include <math.h>


/* work request message when idle */
typedef struct requestmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int from_pe;
/*  int to_rank;  */
  int to_pe;
} requestmsg;

