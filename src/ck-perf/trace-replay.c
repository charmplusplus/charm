#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include "chare.h"
#include "globals.h"
#define MAIN_PERF
#include "trace.h"
#undef MAIN_PERF

#define MAXTRANSSIZE 50000
#define MAXEVENTBUFSIZE	50000
int creation_table[MAXEVENTBUFSIZE];
LOGSTR event_buffer[MAXEVENTBUFSIZE];

#define HASH_TABLE_SIZE 9001
#define HashMap(x) (x*9007 % HASH_TABLE_SIZE)
typedef struct entry {
	int event, destination;
	struct entry *next;
} ENTRY;
ENTRY *hash_table[HASH_TABLE_SIZE];

FILE *fd; 
int (*f)();
LOGSTR *buf;
unsigned int replay_time; 
int type, msg_type, entry, event, dest, pe; 
int read_in_debug_line(), read_in_projections_data(); 

char *pgm;
int CtrRecdTraceMsg = 1;
char *event_file_name;			/* log file name      	*/
int debug_replay, projections_replay;

int event_count=0, translation_count, creation_count;
int  event_index=0, translation_index=0, creation_index=0;
long fevent_position, ftranslation_position, fcreation_position;
int  done_with_event=0,done_with_translation=0,done_with_creation=0;

#define open_and_init(position)  { \
	if ((fd = fopen(event_file_name, "r")) == NULL) \
		CmiPrintf("ERROR: cannot create %s\n",event_file_name); \
 	if (fseek(fd, position, 0) == -1) \
		CmiPrintf("ERROR: cannot seek %s\n",event_file_name); \
	if (debug_replay) f=read_in_debug_line; \
	else f=read_in_projections_data;  \
}

#define close_and_finish(position) { \
	position = ftell(fd); \
	write_event(); \
	fclose(fd); \
}


#define SystemEvent(msg_type) \
	 (!debug_replay && \
	 (msg_type==LdbMsg || msg_type==QdBocMsg || msg_type==QdBroadcastBocMsg))

/**********All the trace functions *****************/
trace_creation(msg_type, entry, envelope)
int msg_type, entry;
ENVELOPE *envelope;
{
        if (msg_type==LdbMsg || msg_type==QdBroadcastBocMsg || msg_type==QdBocMsg)
        {
                SetEnv_event(envelope, -1);
                SetEnv_pe(envelope, -1);
                return;
        }

        SetEnv_event(envelope, determine_creation(msg_type));
        SetEnv_pe(envelope, CmiMyPe());
}


trace_begin_execute(envelope)
ENVELOPE *envelope;
{}


trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{
	if (msg_type==LdbMsg || msg_type==QdBroadcastBocMsg || msg_type==QdBocMsg)
		return;
	increment_event();
}

trace_begin_charminit() 
{
	int *msg;
	ENVELOPE *envelope;
	msg = (int *) CkAllocMsg(sizeof(int));
	envelope = (ENVELOPE *) ENVELOPE_UPTR(msg);
	trace_creation(NewChareMsg, -1, envelope);
}

trace_end_charminit() 
{increment_event();}



trace_enqueue(envelope)
ENVELOPE *envelope;
{}

trace_dequeue(envelope)
ENVELOPE *envelope;
{}


trace_begin_computation()
{}

trace_table(type,tbl,key,pe)
int type,tbl,key,pe;
{}



/***********************************************************************/ 
/*** 	This function is called when the program begins to execute to **/
/*** 	set up the log files.					      **/
/***********************************************************************/ 

log_init()
{ 
	int i;
	int pe;
	int length;
	char temp[1000];
	FILE *fd;
	FILE *translation_file_desc;

	pe = CmiMyPe();

	/* build log file name from pgm name and pe number */
	length = strlen(pgm) + strlen(".") + CmiNumPes() + strlen(".log") + 1;
	event_file_name = (char *) CkAlloc(length);
	sprintf(event_file_name, "%s.%d.rpy", pgm, pe);

	if((fd = fopen(event_file_name, "r")) == NULL)
		printf("ERROR: cannot open %s\n",event_file_name);

	debug_replay=projections_replay=0;
	fscanf(fd, "%s", temp); 
	if (!strcmp(temp, "DEBUG-REPLAY")) debug_replay = 1;
	else if (!strcmp(temp, "PROJECTIONS-REPLAY")) projections_replay = 1;
	else {
		CmiPrintf("ERROR: incorrect replay format for %s\n", event_file_name);
		fclose(fd);
		return;
	}
	fevent_position = ftranslation_position = fcreation_position =  ftell(fd); 
	fclose(fd);

	determine_translation_set(MAXTRANSSIZE);
	determine_event_set(MAXEVENTBUFSIZE);
	if (projections_replay) determine_creation_set(MAXEVENTBUFSIZE);
}



/***********************************************************************/ 
/*** This function is called at the very end to dump the buffers into **/
/*** the log files.						      **/
/***********************************************************************/ 

close_log()
{
}

void PrintStsFile(str)
char *str ;
{ }

/***********************************************************************/ 
/***  This function is used to determine the name of the program.     **/
/***********************************************************************/ 

program_name(s, m)
char *s, *m;
{
	pgm = (char *) malloc(strlen(s));
	strcpy(pgm, s);
}




send_log() {}

CollectTraceFromNodes(msg, data)
char  msg, data;
{}




write_event()
{
/*
	int i;
	char temp[6*10];
	char output[6*MAXEVENTBUFSIZE*10];

       	strcpy(output, "");
	for (i=0; i<event_count; i++)
	{
		strcpy(temp, "");
		sprintf(temp, "[%d] event: %d %d %d %d\n",
			CmiMyPe(),
			event_buffer[i].msg_type, event_buffer[i].entry,
			event_buffer[i].event, event_buffer[i].pe);
		strcat(output, temp);
	}
       	CmiPrintf("%s\n", output);
	CmiPrintf("\n\n\n");
*/
}


write_translation()
{
/*
	int i;
        char temp[4*10];
        char output[4*MAXTRANSSIZE*10];


        strcpy(output, "");
	for (i=0; i<MAXTRANSSIZE; i++)
		if (translation_buffer[i] != -1)
		{
			strcpy(temp, "");
			sprintf(temp, "[%d] trans: %d %d\n",
					CmiMyPe(),  i, translation_buffer[i]);
                	strcat(output, temp);
		}
        CmiPrintf("%s\n", output);
*/
}


increment_event()
{
	int index;
/*
CmiPrintf("[%d] event: event=%d, pe=%d\n", CmiMyPe(), 
event_buffer[event_index].event, event_buffer[event_index].pe);
*/
    event_index++;
	index = event_index%MAXEVENTBUFSIZE;
    if (index>=event_count && !done_with_event)
        determine_event_set(MAXEVENTBUFSIZE);
}

determine_current_event(event, pe)
int *event, *pe;
{
	int index=event_index%MAXEVENTBUFSIZE;

    *event = event_buffer[index].event;
    *pe = event_buffer[index].pe;
    if (event_buffer[index].msg_type==BroadcastBocMsg ||
            event_buffer[index].msg_type==DynamicBocInitMsg)
                *event -= CmiMyPe();
}

determine_creation(msg_type)
int msg_type;
{
	int store;

	if (debug_replay) store=creation_index;
	else store=creation_table[creation_index];

	creation_index++;
 	if (msg_type==BocInitMsg || msg_type==BroadcastBocMsg ||
			msg_type==DynamicBocInitMsg)
		 creation_index += CmiNumPes()-1;

	if (projections_replay)  {
		int index = creation_index%MAXEVENTBUFSIZE;
		if (index>=creation_count && !done_with_creation)
			determine_creation_set(MAXEVENTBUFSIZE);
	}

	return store;
}

determine_translation(env)
ENVELOPE *env;
{
	int x = GetEnv_event(env);

    if (x>=translation_count && !done_with_translation)
        determine_translation_set(MAXTRANSSIZE);
    return DeleteDestination(x);
    TRACE(CmiPrintf("[%d] trans: for %d, dest=%d\n", CmiMyPe(), x, temp));
}


/***********************************************************************/ 
/***  These functions are used to read from the log file.		      **/
/***********************************************************************/ 

determine_event_set(count)
int count;
{
	event_count=0;
	buf = event_buffer;
	open_and_init(fevent_position);
	while (event_count<count && !done_with_event)
	{
		(*f)(fd, 1, &type, &msg_type, &entry, &replay_time, &event, &dest, &pe);

		if (SystemEvent(msg_type)) continue;
		switch (type) {

			case BEGIN_PROCESSING:
                buf->msg_type = msg_type;
				buf->entry = entry;
				buf->event = event;
				buf->pe = pe;
				buf++;
				event_count++;
				break;

			case END_COMPUTATION:
				done_with_event = 1;
				break;

		}
	}
	close_and_finish(fevent_position);
}

determine_translation_set(count)
int count;
{
	open_and_init(ftranslation_position);
	while (translation_count<count && !done_with_translation)
	{
		(*f)(fd, 1, &type, &msg_type, &entry, &replay_time, &event, &dest, &pe);

		if (SystemEvent(msg_type)) continue;
		switch (type) {

			case CREATION:
				if (msg_type == NewChareMsg)  {
					InsertDestination(event, dest);
                    translation_count=event;
				}
				break;

			case END_COMPUTATION:
				done_with_translation = 1;
				break;

		}
	}
	close_and_finish(ftranslation_position);
}

determine_creation_set(count)
int count;
{
	creation_count=0;
	open_and_init(fcreation_position);
	while (creation_count<count && !done_with_creation)
	{
		(*f)(fd, 1, &type, &msg_type, &entry, &replay_time, &event, &dest, &pe);

		if (SystemEvent(msg_type)) continue;
		switch (type) {

			case CREATION:
				creation_table[creation_count++] = event; 
				break;

			case END_COMPUTATION:
				done_with_creation = 1;
				break;

		}
	}
	close_and_finish(fcreation_position);
}


/***********************************************************************/ 
/* Hash table functions.											**/
/***********************************************************************/ 

InsertDestination(event, destination)
int event, destination;
{
    ENTRY *current;
    int index = HashMap(event);

    current = (ENTRY *) CkAlloc(sizeof(ENTRY));
    current->event = event;
    current->destination = destination;
    current->next = hash_table[index];
    hash_table[index] = current;
}


DeleteDestination(event, pe)
int event, pe;
{
    ENTRY *previous = NULL;
    int index = HashMap(event);
    ENTRY *current = hash_table[index];

    while (current != NULL)
        if (current->event==event) {
            if (!previous) hash_table[index] = current->next;
            else previous->next = current->next;
            CkFree(current);
            return current->destination;
        }
        else
            current = current->next;
    printf("*** ERROR *** Cannot determine destination for %d, %d\n",
                pe, event);
    return -1;
}


