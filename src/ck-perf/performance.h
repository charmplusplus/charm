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
 * Revision 2.0  1995-06-02 17:40:29  brunner
 * Reorganized directory structure
 *
 * Revision 1.2  1994/11/11  05:31:17  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:40  brunner
 * Initial revision
 *
 ***************************************************************************/
#define  CREATION           1
#define  BEGIN_PROCESSING   2
#define  END_PROCESSING     3
#define  ENQUEUE            4
#define  DEQUEUE            5
#define  BEGIN_COMPUTATION  6
#define  END_COMPUTATION    7
#define  BEGIN_INTERRUPT    8
#define  END_INTERRUPT      9
#define  INSERT				10
#define  DELETE 			11
#define  FIND			    12


#define  NUMBER_DISPLAYS	7
#define CREATE_NEWCHARE         0
#define CREATE_FORCHARE         1
#define CREATE_FORBOC           2
#define PROCESS_NEWCHARE        3
#define PROCESS_FORCHARE        4
#define PROCESS_FORBOC          5
#define IDLE_TIME               6

#define MAXLOGBUFSIZE  		100000    /* max size of log buffer */
#define MAXLOGMSGSIZE  		400   /* max size of log buffer */

#define INITIAL_TIMESTEP	100

typedef struct _debug {
    int timestep;
    int current_time_interval;
    int display_table[NUMBER_DISPLAYS][MAXLOGMSGSIZE]; 
} PERF_MSG;


#ifdef DEBUGGING_MODE
typedef struct logstr {         /* structure of the log entry 		*/
	int type;		/* creation/processing			*/
	int msg_type;		/* type of message 			*/
	int entry; 		/* entry point message was sent		*/
	int event;		/* message event number			*/
	int pe; 		/* message event processor number	*/
    unsigned int time1;     /* time in microseconds of event        */
} LOGSTR;
#else
typedef struct logstr {         /* structure of the log entry           */
    int type;             /* creation/processing                  */
    int msg_type;         /* type of message                      */
    int entry;              /* entry point message was sent         */
    unsigned int time1;     /* time in microseconds of event        */
} LOGSTR;
#endif


#ifdef MAIN_PERF

#define MAX_COLORS 5
#define MAX_EPS 100
char * colors [] = {
	"black", "blue", "yellow", "red", "green"
};

#endif



