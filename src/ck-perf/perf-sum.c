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
 *      $Log$
 *      Revision 2.12  1997-07-18 19:14:58  milind
 *      Fixed the perfModuleInit call to pass command-line params.
 *      Also added trace_enqueue call to Charm message handler.
 *
 *      Revision 2.11  1997/03/19 23:17:38  milind
 *      Got net-irix to work. Had to modify jsleep to deal with restaring
 *      system calls on interrupts.
 *
 *      Revision 2.10  1995/10/30 22:29:57  sanjeev
 *      converted static variable in CollectPerfFromNodes to Cpv
 *
 * Revision 2.9  1995/10/30  14:31:12  jyelon
 * Fixed an obvious bug, but there's probably still more.
 *
 * Revision 2.8  1995/10/27  21:37:45  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.7  1995/07/27  20:48:27  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/22  23:44:01  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/12  21:36:20  brunner
 * Added prog_name to perfModuleInit(), so argv[0] can be used
 * to generate a unique tace file name.
 *
 * Revision 2.4  1995/07/12  20:23:47  brunner
 * Changed global variable time to now, to avoid conflict with
 * system function time.
 *
 * Revision 2.3  1995/07/10  22:29:40  brunner
 * Created perfModuleInit() to handle CPV macros
 *
 * Revision 2.2  1995/07/06  22:42:54  narain
 * Corrected usage of LdbBocNum to StatisticBocNum
 *
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
/* program to log the trace information */

#include <stdio.h>
#include <string.h>
#include "chare.h"
#include "globals.h"
#define MAIN_PERF
#include "performance.h"
#undef MAIN_PERF
#include "stat.h"

CpvDeclare(char*,pgm);
CpvExtern(int,RecdPerfMsg); 

CpvDeclare(int,display_index);
CpvDeclare(int,last_time_interval);
CpvDeclare(int,current_time_interval);

typedef int display_table_type[NUMBER_DISPLAYS][MAXLOGMSGSIZE];
CpvDeclare(display_table_type,display_table);

CpvDeclare(int,now);
CpvDeclare(int,timestep);
CpvDeclare(int,init_time);
CpvDeclare(int,start_processing_time);
CpvDeclare(int,num_childmsgs);

CpvExtern(int,RecdStatMsg);

perfModuleInit(pargc, argv)
int *pargc;
char **argv;
{
  char nodename[80];

  CpvInitialize(char*,pgm);
  CpvInitialize(int,display_index);
  CpvInitialize(int,last_time_interval);
  CpvInitialize(int,current_time_interval);
  CpvInitialize(display_table_type,display_table);
  CpvInitialize(int,now);
  CpvInitialize(int,timestep);
  CpvInitialize(int,init_time);
  CpvInitialize(int,start_processing_time);
  CpvInitialize(int,num_childmsgs);

  sprintf(nodename,"%d",CmiMyPe());
  program_name(argv[0],nodename);
}

trace_creation(msg_type, entry, envelope)
int msg_type, entry;
ENVELOPE *envelope;
{
	CpvAccess(now) = CkTimer(); 
	adjust_time_interval(CpvAccess(now));
	CpvAccess(display_index) = get_creation_display_index(msg_type); 
	if (CpvAccess(display_index) >=0)
		CpvAccess(display_table)[CpvAccess(display_index)][CpvAccess(current_time_interval)] += 1;
}

trace_begin_execute(envelope)
ENVELOPE *envelope;
{
	int msg_type = GetEnv_msgType(envelope);
	if (((msg_type == BocMsg) || (msg_type == BroadcastBocMsg) ||
               	(msg_type == QdBocMsg) || (msg_type == QdBroadcastBocMsg)) &&
		(GetEnv_EP(envelope) < CsvAccess(NumSysBocEps)))
		return;

	CpvAccess(now) = CkTimer(); 
	adjust_time_interval(CpvAccess(now));
	CpvAccess(start_processing_time) = CpvAccess(now);
	CpvAccess(last_time_interval) = CpvAccess(current_time_interval);
}


trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{
	if (CpvAccess(start_processing_time) == -1)
		return;
	CpvAccess(now) = CkTimer(); 
	adjust_time_interval(CpvAccess(now));
   	CpvAccess(display_index) = get_processing_display_index(msg_type);
	compute_busy(CpvAccess(start_processing_time), CpvAccess(now), CpvAccess(last_time_interval), 
			CpvAccess(current_time_interval));
	if (CpvAccess(display_index) >= 0) 
		update_display(CpvAccess(display_index), CpvAccess(last_time_interval),
				CpvAccess(current_time_interval)); 
	CpvAccess(start_processing_time) = CpvAccess(last_time_interval) = -1;
}

trace_begin_charminit()
{}

trace_end_charminit()
{}

trace_begin_computation()
{}

trace_enqueue(envelope)
ENVELOPE *envelope;
{}

trace_dequeue(envelope)
ENVELOPE *envelope;
{}

trace_table(type,tbl,key,pe)
int type,tbl,key,pe;
{}



/***********************************************************************/ 
/***********************************************************************/ 

adjust_time_interval(time)
unsigned int time;
{
	int temp_time_interval = MAXLOGMSGSIZE - 1;

	CpvAccess(current_time_interval) = (time - CpvAccess(init_time)) / CpvAccess(timestep);
	while (CpvAccess(current_time_interval) >= MAXLOGMSGSIZE)
	{
TRACE(CmiPrintf("[%d] adjust_time_interval: current_time_interval=%d\n", 
CmiMyPe(), CpvAccess(current_time_interval)));

		adjust_timestep(&CpvAccess(timestep), &temp_time_interval,
				CpvAccess(display_table), 2);
		CpvAccess(current_time_interval) = (time - CpvAccess(init_time)) / CpvAccess(timestep);
		if (CpvAccess(start_processing_time) != -1)
			CpvAccess(last_time_interval) = (CpvAccess(start_processing_time) -
						 CpvAccess(init_time)) / CpvAccess(timestep);

TRACE(CmiPrintf("[%d] adjust_time_interval: current_time_interval=%d\n", 
CmiMyPe(), CpvAccess(current_time_interval)));
	}
}


/***********************************************************************/ 
/*** 	This function is called when the program begins to execute to **/
/*** 	set up the log files.					      **/
/***********************************************************************/ 

log_init()
{ 
	int i, j;
	CpvAccess(timestep) = INITIAL_TIMESTEP;

	CpvAccess(RecdPerfMsg) = 0;
	CpvAccess(init_time) = CkTimer();
	CpvAccess(start_processing_time) = -1;
        for (i=0; i<NUMBER_DISPLAYS; i++)
        for (j=0; j<MAXLOGMSGSIZE; j++)
		CpvAccess(display_table)[i][j] = 0;
}


/***********************************************************************/ 
/*** This function is called at the very end to dump the buffers into **/
/*** the log files.						      **/
/***********************************************************************/ 

close_log()
{
	if (CmiMyPe() == 0)
	{
		int i, j; 
		int length;
		FILE *log_file_desc;
		char *log_file_name; 	/* log file name      	*/

		/* build log file name from pgm name and pe number */
		/* flush out the log buffer before closing the log file */
	
		length = strlen(CpvAccess(pgm)) + strlen(".pgm") + 1;
		log_file_name = (char *) CmiAlloc(length);
		sprintf(log_file_name, "%s.log", CpvAccess(pgm));
	
		if((log_file_desc = fopen(log_file_name, "w+")) == NULL)
			printf("*** ERROR *** Cannot Create %s",
					log_file_name);
		fprintf(log_file_desc, "%d %d %d\n", CmiNumPes(),
			CpvAccess(timestep), CpvAccess(current_time_interval)+1);
			

		for (j=0; j<CpvAccess(current_time_interval)+1; j++)
			CpvAccess(display_table)[IDLE_TIME][j] =
				(CpvAccess(display_table)[IDLE_TIME][j]*100)/
				(CmiNumPes()*CpvAccess(timestep));
		for (i=0; i<NUMBER_DISPLAYS; i++)
		{
			for (j=0; j<CpvAccess(current_time_interval)+1; j++)
				fprintf(log_file_desc, "%d ", 
					CpvAccess(display_table)[i][j]); 
			fprintf(log_file_desc, "\n");
		}
	}
}

adjust_timestep(timestep, current_time_interval, display_table, index)
int *timestep;
int *current_time_interval;
int display_table[NUMBER_DISPLAYS][MAXLOGMSGSIZE];
int index;
{
	int i, j, k, l;

TRACE(CmiPrintf("[%d] adjust_timestep: timestep=%d, current_time_interval=%d, index=%d\n",
		CmiMyPe(), *timestep, *current_time_interval, index));

TRACE(print_idle(*current_time_interval));

        *timestep = (*timestep)*index;

        for (i=0; i<NUMBER_DISPLAYS; i++)
        for (j=0,l=0; j<(*current_time_interval+1); j+=index,l++)
	{
                display_table[i][l] = display_table[i][j];
        	for (k=1; k<index; k++)
		if (j+k<(*current_time_interval)+1)
                 	display_table[i][l] += display_table[i][j+k];
	}

	l--;
	/*
	for (j=0; j<l+1; j++)
		display_table[IDLE_TIME][j] /= index;	
	*/
        *current_time_interval = l;

        for (i=0; i<NUMBER_DISPLAYS; i++)
        	for (j=l+1; j<MAXLOGMSGSIZE; j++)
			display_table[i][j] = 0;

TRACE(print_idle(*current_time_interval));
}



add_log(msg)
PERF_MSG *msg;
{
	int i, j;

	if ((msg->timestep != 0) && (msg->timestep != CpvAccess(timestep)))
		if (msg->timestep > CpvAccess(timestep))
			adjust_timestep(&CpvAccess(timestep), &CpvAccess(current_time_interval),
					CpvAccess(display_table),
					(msg->timestep/CpvAccess(timestep)));
		else
			adjust_timestep(&(msg->timestep),
					&(msg->current_time_interval),
					msg->display_table,
					(CpvAccess(timestep)/msg->timestep));

	/* flush out the log buffer before closing the log file */
	for (i=0; i<NUMBER_DISPLAYS; i++)
		for (j=0; j<CpvAccess(current_time_interval)+1; j++)
			CpvAccess(display_table)[i][j] += msg->display_table[i][j];
	/*
	for (j=0; j<CpvAccess(current_time_interval)+1; j++)
		CpvAccess(display_table)[IDLE_TIME][j] /= 2;
	*/

}

/*************************************************************************/ 
/** Send out performance message.					**/
/*************************************************************************/ 
SendOutPerfMsg(mype)
int mype;
{
	int i, j;
	PERF_MSG *msg;

	msg = (PERF_MSG *) CkAllocMsg(sizeof(PERF_MSG));

	msg->current_time_interval = CpvAccess(current_time_interval);
	msg->timestep = CpvAccess(timestep);
	for (i=0; i<NUMBER_DISPLAYS; i++)
	{
	for (j=0; j<CpvAccess(current_time_interval)+1; j++)
		msg->display_table[i][j] = CpvAccess(display_table)[i][j];	
	for (j=CpvAccess(current_time_interval)+1; j<MAXLOGMSGSIZE; j++)
		msg->display_table[i][j] = 0;
	}
TRACE(CmiPrintf("[%d] Send out perf message to %d\n", 
		mype, CmiSpanTreeParent(mype)));

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_PerfCollectNodes), msg,
		CmiSpanTreeParent(mype), BocMsg, StatisticBocNum);
}


/*************************************************************************/ 
/** Adjust log and send out messages if necessary.			**/
/*************************************************************************/ 
send_log()
{
	int mype = CmiMyPe();
	if (CmiNumSpanTreeChildren(mype) == 0)
	{
		CpvAccess(RecdPerfMsg) = 1;
		if (mype != 0)
			SendOutPerfMsg(mype);
	}
}

/************************************************************************/ 
/** Collect performance messages sent from children here.		**/
/************************************************************************/ 
CollectPerfFromNodes(msg, localdataptr)
PERF_MSG *msg;
void  *localdataptr;
{
	int mype;

	CpvAccess(num_childmsgs)++;
	add_log(msg);
	mype = CmiMyPe();

TRACE(CmiPrintf("[%d] CollectPerf..: num_childmsgs=%d,  span=%d\n",
		mype, CpvAccess(num_childmsgs), CmiNumSpanTreeChildren(mype)));
	if (CpvAccess(num_childmsgs) == CmiNumSpanTreeChildren(mype))
	{
		CpvAccess(RecdPerfMsg) = 1;
		if (mype != 0)
			SendOutPerfMsg(mype);
TRACE(CmiPrintf("[%d] RecdStatMsg=%d\n", CmiMyPe(), CpvAccess(RecdStatMsg)));
		if (CpvAccess(RecdStatMsg)) ExitNode(); 	
	}  
}

/***********************************************************************/ 
/***  This function is used to determine the name of the program.     **/
/***********************************************************************/ 

program_name(s, m)
char *s, *m;
{
	CpvAccess(pgm) = (char *) malloc(strlen(s)+1);
	strcpy(CpvAccess(pgm), s);
}





/*************************************************************************/
/** Update the display for this message processing/creation.  		**/
/*************************************************************************/
update_display(index, last_interval, current_interval)
int index;
int last_interval, current_interval;
{
	int i;

	/*
	for (i=last_interval; i<=current_interval; i++)
		(CpvAccess(display_table)[index][i])++;
	*/
	CpvAccess(display_table)[index][last_interval]++;
}



/*************************************************************************/
/** We compute the busy period in this function.			**/
/*************************************************************************/
compute_busy(begin, end, last_interval, current_interval)
int begin, end;
int last_interval, current_interval;
{
	int i;
	int border;

TRACE(CmiPrintf("[%d] compute_busy: begin=%d, end=%d, last_interval=%d, current_interval=%d\n",
		CmiMyPe(), begin, end, last_interval, current_interval));
	if (begin == -1) return;


	if (current_interval > last_interval)
	{
		for (i=last_interval; i<current_interval; i++)
		{
			border = CpvAccess(timestep)*(i+1) + CpvAccess(init_time); 
			CpvAccess(display_table)[IDLE_TIME][i] += border - begin;
				 /* ((border - begin)*100)/CpvAccess(timestep); */
			begin = border;
		}
		CpvAccess(display_table)[IDLE_TIME][current_interval] +=
				 end - border;
				 /* ((end - border)*100)/CpvAccess(timestep); */
	}
	else
		CpvAccess(display_table)[IDLE_TIME][current_interval] +=
			 end - begin;
			 /* ((end - begin)*100)/CpvAccess(timestep); */
}



/*************************************************************************/
/** This function is used to process creation message types.		**/
/*************************************************************************/

get_creation_display_index(msg_type) 
int msg_type; 
{
	int display_index;

    	/*****************************************************************/
	/** Get the proper display table.				**/
    	/******************************************************************/
	switch (msg_type) 
	{
		case NewChareMsg:
			display_index = CREATE_NEWCHARE;
			break;
		case ForChareMsg:
			display_index = CREATE_FORCHARE;
			break;
		case BocMsg:
			display_index = CREATE_FORBOC;
			break;
		default:
			display_index = -1;
			break;
	}
	return display_index;
}



/*************************************************************************/
/*************************************************************************/

get_processing_display_index(msg_type)
int msg_type;
{
	int display_index;

    	/*****************************************************************/
	/** Get the proper display table.				**/
    	/******************************************************************/
	switch (msg_type) 
	{
		case NewChareMsg:
			display_index = PROCESS_NEWCHARE;
			break;
		case ForChareMsg:
			display_index = PROCESS_FORCHARE;
			break;
		case BocMsg:
			display_index = PROCESS_FORBOC;
			break;
		default:
			display_index = -1;
			break;
	}
	return display_index;
}




print_idle(X)
int X;
{
	int j;
	char *str, *string;

	str = (char *) malloc(100);
	string = (char *) malloc(1000);

	CmiPrintf("[%d] print_idle: X=%d, current_time_interval=%d\n",
			 CmiMyPe(), X, CpvAccess(current_time_interval));
	sprintf(string, "");
	for (j=0; j<X+1; j++)
	{
		sprintf(str, "%d ", CpvAccess(display_table)[0][j]);
		strcat(string, str);
	}
	CmiPrintf("[%d] print_idle: %s \n", CmiMyPe(), string);
	
	free(str);
	free(string);
}


void PrintStsFile(str)
char *str ;
{
}
