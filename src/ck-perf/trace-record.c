/* program to log the trace information */

#include <stdio.h>
#include <string.h>
#include "chare.h"
#include "globals.h"
#define MAIN_PERF
#include "trace.h"
#undef MAIN_PERF

char *pgm, *machine;
int RecdTraceMsg = 1;
char *log_file_name;		/* log file name      	*/
LOGSTR logbuf[MAXLOGBUFSIZE];

int  logcnt;        		/* no. of log entries 	*/
int iteration;
int current_event, begin_event, begin_pe;

extern int TotalChares;

FILE *state_file_fd;


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

	if (msg_type==LdbMsg || msg_type==QdBocMsg || msg_type==QdBroadcastBocMsg)
	{
		SetEnv_event(envelope, -1);
		SetEnv_pe(envelope, -1);
		return;
	}

        iteration = 1;
	if (msg_type==BocInitMsg ||
			msg_type==BroadcastBocMsg ||
			msg_type==DynamicBocInitMsg)
				iteration = CmiNumPes(); 

    SetEnv_event(envelope, current_event);
    SetEnv_pe(envelope, CmiMyPe());
	for (i=0; i<iteration; i++)
		add_to_buffer(CREATION, msg_type, entry,  
				GetEnv_event(envelope)+i, GetEnv_pe(envelope));
	current_event += iteration;
}

trace_begin_execute(envelope)
ENVELOPE *envelope;
{
	int msg_type = GetEnv_msgType(envelope);
	int entry = GetEnv_EP(envelope);

	if (msg_type==QdBocMsg || msg_type==QdBroadcastBocMsg ||
			msg_type==LdbMsg) return;
	begin_event = GetEnv_event(envelope);
	if (msg_type==BocInitMsg ||
			msg_type==BroadcastBocMsg ||
			msg_type==DynamicBocInitMsg)
				begin_event += CmiMyPe();
	add_to_buffer(BEGIN_PROCESSING, msg_type, entry, 
					begin_event, GetEnv_pe(envelope));
}


trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{}

trace_begin_charminit()
{
    int *msg;
    ENVELOPE *envelope;
    msg = (int *) CkAllocMsg(sizeof(int));
    envelope = (ENVELOPE *) ENVELOPE_UPTR(msg);
    trace_creation(NewChareMsg, -1, envelope);
	add_to_buffer(BEGIN_PROCESSING, NewChareMsg, -1,
				GetEnv_pe(envelope), GetEnv_event(envelope));
}

trace_end_charminit()
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




trace_begin_computation()
{}

trace_end_computation()
{
	add_to_buffer(END_COMPUTATION, -1, -1, -1, -1);
}


/***********************************************************************/ 
/*** 	Log the event into this processor's buffer, and if full print **/
/*** 	out on the output file.					      **/
/***********************************************************************/ 
add_to_buffer(type, msg_type, entry, event, pe)
int type, msg_type, entry;
int event, pe;
{
	LOGSTR *buf;

	buf  = & (logbuf[logcnt]);
	buf->type 	=  type;
	buf->msg_type 	=  msg_type;
	buf->entry 	=  entry;
        buf->event = event;
        buf->pe = pe; 

	/* write the log into the buffer */
	logcnt++;

	/* if log buffer is full then write out */
	/* the log into log file 		*/
	if (logcnt == MAXLOGBUFSIZE)
		wrtlog(CmiMyPe(), logbuf, MAXLOGBUFSIZE);
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

	/* build log file name from pgm name and pe number */

	length = strlen(pgm) + strlen(".") + CmiNumPes() +
		 strlen(".log") + 1;
	log_file_name = (char *) CkAlloc(length);
	sprintf(log_file_name, "%s.%d.log", pgm, pe);

	if((log_file_desc = fopen(log_file_name, "w+")) == NULL)
		printf("*** ERROR *** Cannot Create %s",log_file_name);
	fprintf(log_file_desc, "DEBUG-RECORD\n");
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
        fprintf(state_file_fd, "PROCESSORS %d\n", CmiNumPes());
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
	pgm = (char *) malloc(strlen(s));
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
        write_out_debug_line(log_file_desc, buf->type, buf->msg_type, 
								buf->entry, -1, buf->event, buf->pe);
		fprintf(log_file_desc, "\n");
	}
	logcnt = 0;
	fflush(log_file_desc);
	fclose(log_file_desc);
}


send_log() {}

CollectTraceFromNodes(msg, data)
char  msg, data;
{}

