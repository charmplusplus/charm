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
 * Revision 2.1  1997-07-07 23:17:33  jyelon
 * Redesigned default-main so there's a C and a C++ version.
 *
 * Revision 2.18  1997/03/24 23:14:02  milind
 * Made Charm-runtime 64-bit safe by removing conversions of pointers to
 * integers. Also, removed charm runtime's dependence of unused argv[]
 * elements being 0. Also, added sim-irix-64 version. It works.
 *
 * Revision 2.17  1997/03/19 04:32:53  jyelon
 * Fixed new ConverseInit
 *
 * Revision 2.16  1997/03/19 04:30:50  jyelon
 * Eliminated all the nonsense pertaining to the SIM version.
 *
 * Revision 2.15  1997/02/13 09:30:37  jyelon
 * Modified default-main for new main structure.
 *
 * Revision 2.14  1996/11/08 22:22:46  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 2.13  1996/07/15 21:03:09  jyelon
 * Changed conv-mach flags from #ifdef to #if
 *
 * Revision 2.12  1996/06/28 21:28:09  jyelon
 * Added special code for simulator version.
 *
 * Revision 2.11  1995/09/19 23:10:24  jyelon
 * added function pointer to 'StartCharm' arglist.
 *
 * Revision 2.10  1995/09/19  17:56:25  sanjeev
 * moved Charm's module inits from user_main to InitializeCharm
 *
 * Revision 2.9  1995/07/19  22:15:24  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/07/12  20:59:58  brunner
 * Added argv[0] to perfModuleInit() call, so performance data files
 * can use the prgram name in the log file name.
 *
 * Revision 2.7  1995/07/10  22:30:49  brunner
 * Added call to perfModuleInit() for CPV macros
 *
 * Revision 2.6  1995/07/03  17:55:55  gursoy
 * changed charm_main to user_main
 *
 * Revision 2.5  1995/06/18  21:55:06  sanjeev
 * removed loop from charm_main, put in CsdScheduler()
 *
 * Revision 2.4  1995/06/13  17:00:16  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/13  14:33:55  gursoy
 * *** empty log message ***
 *
 * Revision 2.2  1995/06/09  16:37:40  gursoy
 * Csv accesses modified
 *
 * Revision 1.4  1995/04/13  20:54:18  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1995/04/02  00:48:53  sanjeev
 * changes for separating Converse
 *
 * Revision 1.2  1995/03/17  23:38:21  sanjeev
 * changes for better message format
 *
 * Revision 1.1  1994/11/18  20:38:11  narain
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include "converse.h"

void charm_init(argc, argv)
int argc;
char **argv;
{
  InitializeCharm(argc, argv);
  StartCharm(argc, argv, (void *)0);
}

main(argc, argv)
int argc;
char *argv[];
{
  ConverseInit(argc, argv, charm_init,0,0);
}
