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
 * Revision 2.12  1997-07-18 19:14:56  milind
 * Fixed the perfModuleInit call to pass command-line params.
 * Also added trace_enqueue call to Charm message handler.
 *
 * Revision 2.11  1997/03/14 20:23:52  milind
 * Made MAXLOGBUFSIZE in projections a commandline parameter.
 * One can now specify it as "+logsize 10000" on the program
 * command line.
 *
 * Revision 2.10  1995/11/05 17:54:33  sanjeev
 * current_event should be initialized to 0, not 1
 *
 * Revision 2.9  1995/11/03  04:34:38  sanjeev
 * for END_PROCESSING, put overload the msg_type field with chare-magic
 *
 * Revision 2.8  1995/10/27  21:37:45  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.7  1995/09/26  19:47:51  sanjeev
 * *** empty log message ***
 *
 * Revision 2.6  1995/09/22  20:44:02  sanjeev
 * bug fixes for working with new runtime
 *
 * Revision 2.5  1995/07/27  20:48:27  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/12  21:36:20  brunner
 * Added prog_name to perfModuleInit(), so argv[0] can be used
 * to generate a unique tace file name.
 *
 * Revision 2.3  1995/07/11  20:34:38  knauff
 * Changed 'uint' to 'un_int' to avoid crashes with gcc v 2.5.8
 * (on the SP, at least)
 *
 * Revision 2.2  1995/07/10  22:29:40  brunner
 * Created perfModuleInit() to handle CPV macros
 *
 * Revision 2.1  1995/06/19  16:47:38  brunner
 * Added Cpv macros and modified to use chareCount instead of TotalChares,
 * pseudoCount instead of TotalPseudos, etc.
 * Doesn't work yet, but I need to get a new copy of the files, so I'm
 * checking it in.
 *
 * Revision 2.0  1995/06/02  17:40:29  brunner
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

CpvDeclare(char*,pgm);

CpvExtern(int,RecdPerfMsg);

CpvDeclare(char*,log_file_name);		/* log file name      	*/
CpvDeclare(int,current_event);

typedef LOGSTR * LOGARR;
CpvDeclare(LOGARR,logbuf);

CpvDeclare(int,logcnt);        		/* no. of log entries 	*/
CpvDeclare(int,iteration);

CpvDeclare(int,store_event);
CpvDeclare(int,store_pe);

typedef unsigned int un_int;
CpvDeclare(un_int, store_time);  /* not used currently? */

CpvDeclare(int,begin_pe);
CpvDeclare(int,begin_event);
CpvDeclare(un_int,begin_processing_time);

CpvDeclare(FILE*,state_file_fd);

CpvExtern(int,chareCount);
CpvExtern(int,pseudoCount);
CpvExtern(int,msgCount);
CpvExtern(int, chareEpsCount);



perfModuleInit(pargc, argv)
int *pargc;
char **argv;
{
  CpvInitialize(char*,pgm);
  CpvInitialize(char*,log_file_name);
  CpvInitialize(int,current_event);
  CpvAccess(current_event) = 0 ;

  CpvInitialize(LOGARR,logbuf);
  CpvInitialize(int,logcnt);        /* no. of log entries 	*/
  CpvInitialize(int,iteration);
  CpvInitialize(int,store_event);
  CpvInitialize(int,store_pe);
  CpvInitialize(un_int, store_time);  /* not used currently? */
  CpvInitialize(int,begin_pe);
  CpvInitialize(int,begin_event);
  CpvInitialize(un_int,begin_processing_time);
  CpvInitialize(FILE*,state_file_fd);

  program_name(argv[0]);
}

void PrintStsFile(str)
char *str ;
{
	fprintf(CpvAccess(state_file_fd),"%s",str) ;
}


/**********All the trace functions *****************/
trace_creation(msg_type, entry, envelope)
int msg_type, entry;
ENVELOPE *envelope;
{
	int i;
	CpvAccess(iteration) = 1;
	if (msg_type == BocInitMsg ||
		msg_type == BroadcastBocMsg || 
		msg_type == QdBroadcastBocMsg ||
		msg_type==DynamicBocInitMsg)
                	CpvAccess(iteration) = CmiNumPes();

	SetEnv_event(envelope, CpvAccess(current_event));
	SetEnv_pe(envelope, CmiMyPe());
	for (i=0; i<CpvAccess(iteration); i++)
		add_to_buffer(CREATION, msg_type, entry, CkUTimer(), 
					GetEnv_event(envelope)+i, GetEnv_pe(envelope));
	CpvAccess(current_event) += CpvAccess(iteration);	
}

trace_begin_execute(envelope)
ENVELOPE *envelope;
{
	int msg_type = GetEnv_msgType(envelope);
	CpvAccess(begin_event) = GetEnv_event(envelope);
	if (msg_type ==  BocInitMsg ||
		msg_type == BroadcastBocMsg || 
		msg_type == QdBroadcastBocMsg ||
		msg_type==DynamicBocInitMsg)
			CpvAccess(begin_event) += CmiMyPe();
	CpvAccess(begin_pe) = GetEnv_pe(envelope);
	add_to_buffer(BEGIN_PROCESSING, msg_type, GetEnv_EP(envelope), CkUTimer(),
				 CpvAccess(begin_event), CpvAccess(begin_pe));
}

trace_end_execute(id, msg_type, entry)
int id, msg_type, entry;
{
	/* Overload the msg_type field : put the id (magic number) into it */
	add_to_buffer(END_PROCESSING, id, entry, CkUTimer(),
						CpvAccess(begin_event), CpvAccess(begin_pe));
}

trace_begin_charminit() 
{
    int *msg;
    ENVELOPE *envelope;
    msg = (int *) CkAllocMsg(sizeof(int));
    envelope = (ENVELOPE *) ENVELOPE_UPTR(msg);
    trace_creation(NewChareMsg, -1, envelope);
    CpvAccess(store_pe) = GetEnv_pe(envelope);
    CpvAccess(store_event) = GetEnv_event(envelope);
	add_to_buffer(BEGIN_PROCESSING, NewChareMsg, -1, CkUTimer(),
					CpvAccess(store_event), CpvAccess(store_pe));

}

trace_end_charminit() 
{
    add_to_buffer(END_PROCESSING, NewChareMsg, -1, CkUTimer(),
		  CpvAccess(store_event), CpvAccess(store_pe));
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
CmiMyPe(), CpvAccess(logcnt), type, msg_type, entry, t1, t2, event));
	buf  = & (CpvAccess(logbuf)[CpvAccess(logcnt)]);
	buf->type 	=  type;
	buf->msg_type 	=  msg_type;
	buf->entry 	=  entry;
	buf->time1 = t1;
	buf->event = event;
	buf->pe = pe;

	/* write the log into the buffer */
	CpvAccess(logcnt)++;

	/* if log buffer is full then write out */
	/* the log into log file 		*/
	if (CpvAccess(logcnt) == CpvAccess(LogBufSize))
	{
		int begin_interrupt;

		begin_interrupt = CkUTimer();
		wrtlog(CmiMyPe(), CpvAccess(logbuf), CpvAccess(LogBufSize));

		buf = &(CpvAccess(logbuf)[CpvAccess(logcnt)]);
		buf->type = BEGIN_INTERRUPT;
		buf->time1 = begin_interrupt;
		buf->event = CpvAccess(current_event);
		buf->pe = CmiMyPe();
		CpvAccess(logcnt)++;
		buf++;

		buf->type = END_INTERRUPT;
		buf->time1 = CkUTimer();
		buf->event = CpvAccess(current_event)++;
		buf->pe = CmiMyPe();
		CpvAccess(logcnt)++;
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


	CpvAccess(RecdPerfMsg)=1;
	CpvAccess(current_event)=0;
	CpvAccess(begin_pe)=-1;
	CpvAccess(begin_event)=-1;
	CpvAccess(begin_processing_time)=-1;
        CpvAccess(logbuf) = 
           (LOGARR) CmiAlloc(sizeof(LOGSTR)*CpvAccess(LogBufSize));

	pe = CmiMyPe();

	/* build log file name from pgm name and pe number */

	length = strlen(CpvAccess(pgm)) + strlen(".") + CmiNumPes() +
		 strlen(".log") + 1;
	CpvAccess(log_file_name) = (char *) CmiAlloc(length);
	sprintf(CpvAccess(log_file_name), "%s.%d.log", CpvAccess(pgm), pe);

	if((log_file_desc = fopen(CpvAccess(log_file_name), "w+")) == NULL)
		printf("*** ERROR *** Cannot Create %s",CpvAccess(log_file_name));
	fprintf(log_file_desc, "PROJECTIONS-RECORD\n");
	fclose(log_file_desc);
	CpvAccess(logcnt) = 0; 
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
	buf = CpvAccess(logbuf);
    	trace_end_computation();
	if (CpvAccess(logcnt))
		wrtlog(pe, buf, CpvAccess(logcnt));
	if (pe == 0)
	{
		char *state_file;

		state_file = (char *) malloc(strlen(CpvAccess(pgm)) + strlen(".sts") + 1);
		strcpy(state_file, CpvAccess(pgm));
		strcat(state_file, ".sts");
		CpvAccess(state_file_fd) = (FILE *) fopen(state_file, "w");
		
		fprintf(CpvAccess(state_file_fd), "MACHINE %s\n",CMK_MACHINE_NAME );
		fprintf(CpvAccess(state_file_fd), "PROCESSORS %d\n", CmiNumPes());
		fprintf(CpvAccess(state_file_fd), "TOTAL_CHARES %d\n", 
													CpvAccess(chareCount));
		fprintf(CpvAccess(state_file_fd), "TOTAL_EPS %d\n", 
					   	  CpvAccess(chareEpsCount)+1);   

		fprintf(CpvAccess(state_file_fd), "TOTAL_MSGS %d\n", 
												CpvAccess(msgCount));
		fprintf(CpvAccess(state_file_fd), "TOTAL_PSEUDOS %d\n", 
												CpvAccess(pseudoCount));

		/* first 3 chares are NULLCHARE, CkChare_ACC, CkChare_MONO */
		for (i=0; i<CpvAccess(chareCount); i++)
			fprintf(CpvAccess(state_file_fd), "CHARE %d %s\n", i, 
										CsvAccess(ChareNamesTable)[i]);

   		for (i=CsvAccess(NumSysBocEps); i<CpvAccess(chareEpsCount); i++) {
		    if ( CsvAccess(EpInfoTable)[i].chare_or_boc == CHARE ) {
       			fprintf(CpvAccess(state_file_fd), "ENTRY CHARE %d %s %d %d\n",
               			i, CsvAccess(EpInfoTable)[i].name, 
						CsvAccess(EpInfoTable)[i].chareindex,
						CsvAccess(EpInfoTable)[i].messageindex);
		    }
		    else {
       			fprintf(CpvAccess(state_file_fd), "ENTRY BOC %d %s %d %d\n",
               		i, CsvAccess(EpInfoTable)[i].name, 
					CsvAccess(EpInfoTable)[i].chareindex,
					CsvAccess(EpInfoTable)[i].messageindex);
		    }
		}

   		for (i=0; i<CpvAccess(msgCount); i++)
   			fprintf(CpvAccess(state_file_fd), "MESSAGE %d %d\n",
               					i, CsvAccess(MsgToStructTable)[i].size);

   		for (i=0; i<CpvAccess(pseudoCount); i++)
   			fprintf(CpvAccess(state_file_fd), "PSEUDO %d %d %s\n",
               					i, CsvAccess(PseudoTable)[i].type,
								CsvAccess(PseudoTable)[i].name);


   		fprintf(CpvAccess(state_file_fd), "END\n");
		fflush(CpvAccess(state_file_fd));
		fclose(CpvAccess(state_file_fd));
	}
        CmiFree(CpvAccess(logbuf));
}


/***********************************************************************/ 
/***  This function is used to determine the name of the program.     **/
/***********************************************************************/ 
program_name(s)
char *s;
{
	CpvAccess(pgm) = (char *) malloc(strlen(s) + 1);
	strcpy(CpvAccess(pgm), s);
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

	if((log_file_desc = fopen(CpvAccess(log_file_name), "a")) == NULL)
		CmiPrintf("*** ERROR *** Cannot Create %s",
			  CpvAccess(log_file_name));
	for (i=0; i<count; i++, buf++)
	{
		write_out_projections_line(log_file_desc, 
								buf->type, buf->msg_type,
								buf->entry, buf->time1,
								buf->event, buf->pe);
		fprintf(log_file_desc, "\n");
	}
	CpvAccess(logcnt) = 0;
	fflush(log_file_desc);
	fclose(log_file_desc);
}



send_log() {}

CollectPerfFromNodes(msg, data)
char  msg, data;
{}

