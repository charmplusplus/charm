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

char *pgm;
int RecdPerfMsg; 
char *log_file_name; 	/* log file name      	*/

int display_index;
int last_time_interval, current_time_interval;
int display_table[NUMBER_DISPLAYS][MAXLOGMSGSIZE];
int time, timestep, init_time, start_processing_time;

extern int RecdStatMsg;

trace_creation(msg_type, entry, envelope)
int msg_type, entry;
ENVELOPE *envelope;
{
	time = CkTimer(); 
	adjust_time_interval(time);
	display_index = get_creation_display_index(msg_type); 
	if (display_index >=0)
		display_table[display_index][current_time_interval] += 1;
}

trace_begin_execute(envelope)
ENVELOPE *envelope;
{
	int msg_type = GetEnv_msgType(envelope);
	if (((msg_type == BocMsg) || (msg_type == BroadcastBocMsg) ||
               	(msg_type == QdBocMsg) || (msg_type == QdBroadcastBocMsg)) &&
		(GetEnv_EP(envelope) < NumSysBocEps))
		return;

	time = CkTimer(); 
	adjust_time_interval(time);
	start_processing_time = time;
	last_time_interval = current_time_interval;
}


trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{
	if (start_processing_time == -1)
		return;
	time = CkTimer(); 
	adjust_time_interval(time);
   	display_index = get_processing_display_index(msg_type);
	compute_busy(start_processing_time, time, last_time_interval, 
			current_time_interval);
	if (display_index >= 0) 
		update_display(display_index, last_time_interval,
				current_time_interval); 
	start_processing_time = last_time_interval = -1;
}

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

	current_time_interval = (time - init_time) / timestep;
	while (current_time_interval >= MAXLOGMSGSIZE)
	{
TRACE(CmiPrintf("[%d] adjust_time_interval: current_time_interval=%d\n", 
CmiMyPe(), current_time_interval));

		adjust_timestep(&timestep, &temp_time_interval,
				display_table, 2);
		current_time_interval = (time - init_time) / timestep;
		if (start_processing_time != -1)
			last_time_interval = (start_processing_time -
						 init_time) / timestep;

TRACE(CmiPrintf("[%d] adjust_time_interval: current_time_interval=%d\n", 
CmiMyPe(), current_time_interval));
	}
}


/***********************************************************************/ 
/*** 	This function is called when the program begins to execute to **/
/*** 	set up the log files.					      **/
/***********************************************************************/ 

log_init()
{ 
	int i, j;
	timestep = INITIAL_TIMESTEP;


	RecdPerfMsg = 0;
	init_time = CkTimer();
	start_processing_time = -1;
        for (i=0; i<NUMBER_DISPLAYS; i++)
        for (j=0; j<MAXLOGMSGSIZE; j++)
		display_table[i][j] = 0;
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

		/* build log file name from pgm name and pe number */
		/* flush out the log buffer before closing the log file */
	
		length = strlen(pgm) + strlen(".pgm") + 1;
		log_file_name = (char *) CmiAlloc(length);
		sprintf(log_file_name, "%s.log", pgm);
	
		if((log_file_desc = fopen(log_file_name, "w+")) == NULL)
			printf("*** ERROR *** Cannot Create %s",
					log_file_name);
		fprintf(log_file_desc, "%d %d %d\n", CmiNumPe(),
			timestep, current_time_interval+1);
			

		for (j=0; j<current_time_interval+1; j++)
			display_table[IDLE_TIME][j] =
				(display_table[IDLE_TIME][j]*100)/
				(CmiNumPe()*timestep);
		for (i=0; i<NUMBER_DISPLAYS; i++)
		{
			for (j=0; j<current_time_interval+1; j++)
				fprintf(log_file_desc, "%d ", 
					display_table[i][j]); 
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

	if ((msg->timestep != 0) && (msg->timestep != timestep))
		if (msg->timestep > timestep)
			adjust_timestep(&timestep, &current_time_interval,
					display_table,
					(msg->timestep/timestep));
		else
			adjust_timestep(&(msg->timestep),
					&(msg->current_time_interval),
					msg->display_table,
					(timestep/msg->timestep));

	/* flush out the log buffer before closing the log file */
	for (i=0; i<NUMBER_DISPLAYS; i++)
		for (j=0; j<current_time_interval+1; j++)
			display_table[i][j] += msg->display_table[i][j];
	/*
	for (j=0; j<current_time_interval+1; j++)
		display_table[IDLE_TIME][j] /= 2;
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

	msg->current_time_interval = current_time_interval;
	msg->timestep = timestep;
	for (i=0; i<NUMBER_DISPLAYS; i++)
	{
	for (j=0; j<current_time_interval+1; j++)
		msg->display_table[i][j] = display_table[i][j];	
	for (j=current_time_interval+1; j<MAXLOGMSGSIZE; j++)
		msg->display_table[i][j] = 0;
	}
TRACE(CmiPrintf("[%d] Send out perf message to %d\n", 
		mype, CmiSpanTreeParent(mype)));

	GeneralSendMsgBranch(StatPerfCollectNodes_EP, msg,
		CmiSpanTreeParent(mype), USERcat, BocMsg, LdbBocNum);
}


/*************************************************************************/ 
/** Adjust log and send out messages if necessary.			**/
/*************************************************************************/ 
send_log()
{
	int mype = CmiMyPe();
	if (CmiNumSpanTreeChildren(mype) == 0)
	{
		RecdPerfMsg = 1;
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
	static int recd = 0;

	recd++;
	add_log(msg);
	mype = CmiMyPe();

TRACE(CmiPrintf("[%d] CollectPerf..: recd=%d,  span=%d\n",
		mype, recd, CmiNumSpanTreeChildren(mype)));
	if (recd == CmiNumSpanTreeChildren(mype))
	{
		RecdPerfMsg = 1;
		if (mype != 0)
			SendOutPerfMsg(mype);
TRACE(CmiPrintf("[%d] RecdStatMsg=%d\n", CmiMyPe(), RecdStatMsg));
		if (RecdStatMsg) ExitNode(); 	
	}  
}

/***********************************************************************/ 
/***  This function is used to determine the name of the program.     **/
/***********************************************************************/ 

program_name(s, m)
char *s, *m;
{
	pgm = (char *) malloc(strlen(s)+1);
	strcpy(pgm, s);
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
		(display_table[index][i])++;
	*/
	display_table[index][last_interval]++;
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
			border = timestep*(i+1) + init_time; 
			display_table[IDLE_TIME][i] += border - begin;
				 /* ((border - begin)*100)/timestep; */
			begin = border;
		}
		display_table[IDLE_TIME][current_interval] +=
				 end - border;
				 /* ((end - border)*100)/timestep; */
	}
	else
		display_table[IDLE_TIME][current_interval] +=
			 end - begin;
			 /* ((end - begin)*100)/timestep; */
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
			 CmiMyPe(), X, current_time_interval);
	sprintf(string, "");
	for (j=0; j<X+1; j++)
	{
		sprintf(str, "%d ", display_table[0][j]);
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
