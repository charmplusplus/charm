/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.3  1995/05/04  22:11:10  jyelon
 * *** empty log message ***
 *
 * Revision 1.2  1994/11/11  05:31:06  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:28  brunner
 * Initial revision
 *
 ***************************************************************************/
#define MAXMEMSTAT 10 
/* MAXMEMSTAT is also defined in memory management */

extern int CstatsMaxChareQueueLength;
extern int CstatsMaxForChareQueueLength;
extern int CstatsMaxFixedChareQueueLength;
extern int MemStatistics[];


typedef struct message3 {
    int srcPE;
    int chareQueueLength;
    int forChareQueueLength;
    int fixedChareQueueLength;
    int charesCreated;
    int charesProcessed;
    int forCharesCreated;
    int forCharesProcessed;
    int bocMsgsCreated;
    int bocMsgsProcessed;
    int nodeMemStat[MAXMEMSTAT];
} STAT_MSG;


typedef struct dummy_message {
    int dummy;
} DUMMY_STAT_MSG;

