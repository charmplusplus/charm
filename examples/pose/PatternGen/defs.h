#ifndef _DEFS_H
#define _DEFS_H

#include <unistd.h>
#include <math.h>
#include <time.h>

#define NUM_AVAILABLE_PATTERNS 4

// set to "if (1) CkPrintf" to turn on debug print statements
#define dbPrintf if (0) CkPrintf

// for pattern 0
#define P0_MESSAGES_TO_SEND 20
#define P0_MSG_TRANSIT_TIME 100
#define P0_ELAPSE_TIME 50

// for pattern 1
#define P1_BASE_MSG_TRANSIT_TIME 80
#define P1_MSG_TRANSIT_TIME_RANGE 41  // gives range of 0-40
#define P1_ITERS 3
#define P1_MESSAGES_PER_ITER 5
#define P1_SHORT_DELAY 200
#define P1_LARGE_DELAY 1000000

// for pattern 2
#define P2_MESSAGES_TO_SEND 100
#define P2_MSG_TRANSIT_TIME 200
#define P2_ELAPSE_TIME 800

// for pattern 3
#define P3_MESSAGES_TO_SEND 100
#define P3_MSG_TRANSIT_TIME 1000
#define P3_ELAPSE_TIME 800

#endif  // _DEFS_H
