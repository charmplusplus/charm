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
 * Revision 1.3  1995/04/13  20:55:09  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/12/02  00:02:37  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:40:02  brunner
 * Initial revision
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

char *pgm, *machine;
int RecdPerfMsg = 1;
char *log_file_name;		/* log file name      	*/
int current_event = 0;
LOGSTR logbuf[MAXLOGBUFSIZE];

int  logcnt;        		/* no. of log entries 	*/
int iteration;

int store_event, store_pe;
unsigned int store_time;

int begin_pe = -1, begin_event = -1;
unsigned int begin_processing_time = -1;


FILE *state_file_fd;

extern int TotalChares;


void PrintStsFile(str)
char *str ;
{
	fprintf(state_file_fd,"%s",str) ;
}


/**********All the trace functions *****************/
trace_creation(msg_type, entry, envelope)
int msg_type, entry;
ENVELOPE *envelope;
{
	int i;
	iteration = 1;
	if (msg_type == BocInitMsg ||
		msg_type == BroadcastBocMsg || 
		msg_type == QdBroadcastBocMsg ||
		msg_type==DynamicBocInitMsg)
                	iteration = CmiNumPe();

	SetEnv_event(envelope, current_event);
	SetEnv_pe(envelope, CmiMyPe());
	for (i=0; i<iteration; i++)
		add_to_buffer(CREATION, msg_type, entry, CkUTimer(), 
					GetEnv_event(envelope)+i, GetEnv_pe(envelope));
	current_event += iteration;	
}

trace_begin_execute(envelope)
ENVELOPE *envelope;
{
	int msg_type = GetEnv_msgType(envelope);
	begin_event = GetEnv_event(envelope);
	if (msg_type ==  BocInitMsg ||
		msg_type == BroadcastBocMsg || 
		msg_type == QdBroadcastBocMsg ||
		msg_type==DynamicBocInitMsg)
			begin_event += CmiMyPe();
	begin_pe = GetEnv_pe(envelope);
	add_to_buffer(BEGIN_PROCESSING, msg_type, GetEnv_EP(envelope), CkUTimer(),
				 begin_event, begin_pe);
}

trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{
	add_to_buffer(END_PROCESSING, msg_type, entry, CkUTimer(),
						begin_event, begin_pe);
}

trace_begin_charminit() 
{
    int *msg;
    ENVELOPE *envelope;
    msg = (int *) CkAllocMsg(sizeof(int));
    envelope = (ENVELOPE *) ENVELOPE_UPTR(msg);
    trace_creation(NewChareMsg, -1, envelope);
    store_pe = GetEnv_pe(envelope);
    store_event = GetEnv_event(envelope);
	add_to_buffer(BEGIN_PROCESSING, NewChareMsg, -1, CkUTimer(),
					store_event, store_pe);

}

trace_end_charminit() 
{
    add_to_buffer(END_PROCESSING, NewChareMsg, -1, CkUTimer(),
					store_event, store_pe);
}


trace_enqueue(envelope)
ENVELOPE *envelope;
{
	int add=0;
	int msg_type = GetEnv_msgType(envelope);
    if (msg_type ==  BocInitMsg || msg_type == BroadcastBocMsg ||
        msg_type == QdBroadcastBocMsg || msg_type==DynamicBocInitMsg)
            add = CmiMyPe();

	add_to_buffer(ENQUEUE, GetEnv_msgType(envelope), GetEnv_EP(envelope),
					CkUTimer(), 
					GetEnv_event(envelope)+add, GetEnv_pe(envelope));
}

trace_dequeue(envelope)
ENVELOPE *envelope;
{
	int add=0;
	int msg_type = GetEnv_msgType(envelope);
    if (msg_type ==  BocInitMsg || msg_type == BroadcastBocMsg ||
        msg_type == QdBroadcastBocMsg || msg_type==DynamicBocInitMsg)
            add = CmiMyPe();

	add_to_buffer(DEQUEUE, GetEnv_msgType(envelope), GetEnv_EP(envelope),
					CkUTimer(), 
					GetEnv_event(envelope), GetEnv_pe(envelope));
}

trace_table(type, tbl, key, pe)
int type, tbl, key, pe;
{
	add_to_buffer(type, tbl, key, CkUTimer(), -1, pe);
}

trace_begin_computation()
{
	add_to_buffer(BEGIN_COMPUTATION, -1, -1, CkUTimer(), -1, -1);
}

trace_end_computation()
{
	add_to_buffer(END_COMPUTATION, -1, -1, CkUTimer(), -1, -1, -1);
}

/***********************************************************************/ 
/*** 	Log the event into this processor's buffer, and if full print **/
/*** 	out on the output file.					      **/
/***********************************************************************/ 

add_to_buffer(type, msg_type, entry, t1, event, pe)
int type;
int msg_type;
int entry;
unsigned int t1;
int event, pe;
{
	LOGSTR *buf;

TRACE(CmiPrintf("[%d] add: cnt=%d, type=%d, msg=%d, ep=%d, t1=%d, t2=%d, event=%d\n",
CmiMyPe(), logcnt, type, msg_type, entry, t1, t2, event));
	buf  = & (logbuf[logcnt]);
	buf->type 	=  type;
	buf->msg_type 	=  msg_type;
	buf->entry 	=  entry;
	buf->time1 = t1;
	buf->event = event;
	buf->pe = pe;

	/* write the log into the buffer */
	logcnt++;

	/* if log buffer is full then write out */
	/* the log into log file 		*/
	if (logcnt == MAXLOGBUFSIZE)
	{
		int begin_interrupt;

		begin_interrupt = CkUTimer();
		wrtlog(CmiMyPe(), logbuf, MAXLOGBUFSIZE);

		buf = &(logbuf[logcnt]);
		buf->type = BEGIN_INTERRUPT;
		buf->time1 = begin_interrupt;
		buf->event = current_event;
		buf->pe = CmiMyPe();
		logcnt++;
		buf++;

		buf->type = END_INTERRUPT;
		buf->time1 = CkUTimer();
		buf->event = current_event++;
		buf->pe = CmiMyPe();
		logcnt++;
	}
}


/***********************************************************************/ 
/*** 	This function is called when the program begins to execute to **/
/*** 	set up the log files.					      **/
/***********************************************************************/ 

log_init()
{ 
	int pe;
	int length;
	FILE *log_file_desc;

	pe = CmiMyPe();
 	begin_processing_time = -1;

	/* build log file name from pgm name and pe number */

	length = strlen(pgm) + strlen(".") + CmiNumPe() +
		 strlen(".log") + 1;
	log_file_name = (char *) CkAlloc(length);
	sprintf(log_file_name, "%s.%d.log", pgm, pe);

	if((log_file_desc = fopen(log_file_name, "w+")) == NULL)
		printf("*** ERROR *** Cannot Create %s",log_file_name);
	fprintf(log_file_desc, "PROJECTIONS-RECORD\n");
	fclose(log_file_desc);
	logcnt = 0; 
}


/***********************************************************************/ 
/*** This function is called at the very end to dump the buffers into **/
/*** the log files.						      **/
/***********************************************************************/ 

close_log()
{
	int i;
	int pe;
	LOGSTR *buf;	


	pe = CmiMyPe();

	/* flush out the log buffer before closing the log file */
	buf = logbuf;
    	trace_end_computation();
	if (logcnt)
		wrtlog(pe, buf, logcnt);
	if (pe == 0)
	{
		char *state_file;

		state_file = (char *) malloc(strlen(pgm) + strlen(".sts") + 1);
		strcpy(state_file, pgm);
		strcat(state_file, ".sts");
		state_file_fd = (FILE *) fopen(state_file, "w");
		

		fprintf(state_file_fd, "MACHINE %s\n", machine);
		fprintf(state_file_fd, "PROCESSORS %d\n", CmiNumPe());
		fprintf(state_file_fd, "TOTAL_CHARES %d\n", TotalChares);
		fprintf(state_file_fd, "TOTAL_EPS %d\n", 
		   /*	  TotalEps+TotalBocEps+1);   */
		   	  TotalEps+1);   

		fprintf(state_file_fd, "TOTAL_MSGS %d\n", TotalMsgs);
		fprintf(state_file_fd, "TOTAL_PSEUDOS %d\n", TotalPseudos-1);

        	for (i=0; i<TotalChares-1; i++)
           		fprintf(state_file_fd, "CHARE %d %s\n",
			 	i, ChareNamesTable[i]);

       		for (i=NumSysBocEps; i<TotalEps; i++) {
		    if ( EpChareTypeTable[i] == CHARE ) {
           			fprintf(state_file_fd, "ENTRY CHARE %d %s %d %d\n",
               			i, EpNameTable[i], EpChareTable[i],
						EpToMsgTable[i]);
		    }
		    else {
           			fprintf(state_file_fd, "ENTRY BOC %d %s %d %d\n",
               		i, EpNameTable[i], EpChareTable[i],
					EpToMsgTable[i]);
		    }
		}

       		for (i=0; i<TotalMsgs; i++)
           			fprintf(state_file_fd, "MESSAGE %d %d\n",
               		i, MsgToStructTable[i].size);

       		for (i=0; i<TotalPseudos-1; i++)
           			fprintf(state_file_fd, "PSEUDO %d %d %s\n",
               			i, PseudoTable[i].type,
						PseudoTable[i].name);


         	fprintf(state_file_fd, "END\n");
		fflush(state_file_fd);
		fclose(state_file_fd);
	}
}


/***********************************************************************/ 
/***  This function is used to determine the name of the program.     **/
/***********************************************************************/ 

program_name(s, m)
char *s, *m;
{
	pgm = (char *) malloc(strlen(s) + 1);
	strcpy(pgm, s);
	machine = (char *) malloc(strlen(m) + 1);
	strcpy(machine, m);
}


/***********************************************************************/ 
/***  This function is used to write into the log file, when the      **/
/***  buffer is full, or the program has terminated.		      **/
/***********************************************************************/ 

wrtlog(pe, buffer, count)
int pe;
LOGSTR *buffer;
int count;
{
	int i;
	LOGSTR *buf; 
	FILE *log_file_desc;

	buf = buffer;

	/* open the logfile in append mode, write the log buffer and close it */

	if((log_file_desc = fopen(log_file_name, "a")) == NULL)
		CmiPrintf("*** ERROR *** Cannot Create %s",log_file_name);
	for (i=0; i<count; i++, buf++)
	{
		write_out_projections_line(log_file_desc, 
								buf->type, buf->msg_type,
								buf->entry, buf->time1,
								buf->event, buf->pe);
		fprintf(log_file_desc, "\n");
	}
	logcnt = 0;
	fflush(log_file_desc);
	fclose(log_file_desc);
}



send_log() {}

CollectPerfFromNodes(msg, data)
char  msg, data;
{}

