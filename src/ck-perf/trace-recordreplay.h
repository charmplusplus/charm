/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef _RECORDREPLAY_H
#define _RECORDREPLAY_H

#include <stdio.h>
#include <errno.h>

#include "trace.h"
#include "envelope.h"
#include "register.h"
#include "trace-common.h"

// initial bin size, time in seconds
#define  BIN_SIZE	0.001

#define  MAX_MARKS       256

#define  MAX_PHASES       10


/// class for recording trace record replay 
/**
  TraceRecordReplay increments curEvent variable in the envelope everytime a message is executed 
*/
class TraceRecordReplay : public Trace {
    int curevent;
  public:
    TraceRecordReplay(char **argv);
    void creation(envelope *e, int epIdx, int num=1);

    void beginExecute(envelope *e);
};

#endif

/*@}*/
