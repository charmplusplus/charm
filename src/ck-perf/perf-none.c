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
 * Revision 1.2  1995/04/02  00:49:19  sanjeev
 * changes for separating Converse
 *
 * Revision 1.1  1994/11/03  17:40:00  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

/*
#include "chare.h"
*/

int RecdPerfMsg = 1;

program_name(s,m) char *s, *m; {}

log_init(){}

trace_creation(msg_type,entry,envelope)
int msg_type, entry;
void *envelope;
{}

trace_begin_execute(envelope)
void *envelope;
{}

trace_end_execute(id,msg_type,entry)
int id, msg_type, entry;
{}

trace_enqueue(envelope)
void *envelope;
{}

trace_dequeue(envelope)
void *envelope;
{}

trace_table(type,tbl,key,pe)
int type,tbl,key,pe;
{}


send_log() {}

CollectPerfFromNodes(msg, data)
char msg, data;
{}

close_log(){}

void PrintStsFile(str)
char *str ;
{
}
