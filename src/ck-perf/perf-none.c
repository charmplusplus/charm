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
 * Revision 2.6  1997-03-14 20:23:51  milind
 * Made MAXLOGBUFSIZE in projections a commandline parameter.
 * One can now specify it as "+logsize 10000" on the program
 * command line.
 *
 * Revision 2.5  1995/07/22 23:44:01  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/12  21:36:20  brunner
 * Added prog_name to perfModuleInit(), so argv[0] can be used
 * to generate a unique tace file name.
 *
 * Revision 2.3  1995/07/11  16:48:17  gursoy
 * added void to perfModuleInit
 *
 * Revision 2.2  1995/07/10  22:29:40  brunner
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

void perfModuleInit(prog_name) char *prog_name; {}

program_name(s,m) char *s, *m; {}

log_init() {CpvAccess(RecdPerfMsg) = 1;}

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

trace_begin_charminit()
{}

trace_end_charminit()
{}

trace_begin_computation()
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
