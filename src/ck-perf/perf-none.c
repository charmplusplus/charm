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
 * Revision 2.2  1995-07-10 22:29:40  brunner
 * Created perfModuleInit() to handle CPV macros
 *
 * Revision 2.1  1995/06/08  17:18:11  gursoy
 * Cpv macro changes done
 *
 * Revision 1.2  1995/04/02  00:49:19  sanjeev
 * changes for separating Converse
 *
 * Revision 1.1  1994/11/03  17:40:00  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include "chare.h"

CpvExtern(int, RecdPerfMsg);

perfModuleInit() {}

program_name(s,m) char *s, *m; {}

log_init(){CpvAccess(RecdPerfMsg) = 1;}

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
