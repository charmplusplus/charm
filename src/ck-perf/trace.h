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
#define  USER_EVENT                 13
#define  BEGIN_IDLE		    14
#define  END_IDLE		    15
#define  BEGIN_PACK	16
#define  END_PACK	17
#define  BEGIN_UNPACK	18
#define  END_UNPACK	19


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


typedef struct logstr {         /* structure of the log entry 		*/
	int type;		/* creation/processing			*/
	int msg_type;		/* type of message 			*/
	int entry; 		/* entry point message was sent		*/
	int event;		/* message event number			*/
	int pe; 		/* message event processor number	*/
    unsigned int time1;     /* time in microseconds of event        */
} LOGSTR;

CpvExtern(int, traceOn);

#ifdef MAIN_PERF

#define MAX_COLORS 5
#define MAX_EPS 100
char * colors [] = {
	"black", "blue", "yellow", "red", "green"
};

#endif



