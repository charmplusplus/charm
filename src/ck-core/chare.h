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
 * Revision 2.2  1997-07-26 16:41:09  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08 17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.8  1995/04/23  21:23:15  sanjeev
 * *** empty log message ***
 *
 * Revision 1.7  1995/04/23  21:18:53  sanjeev
 * put #include converse.h
 *
 * Revision 1.6  1995/04/23  00:52:14  sanjeev
 * moved #define _CK_VARSIZE_UNIT 8 from conv-mach.h to chare.h
 *
 * Revision 1.5  1995/04/02  00:48:28  sanjeev
 * changes for separating Converse
 *
 * Revision 1.4  1995/03/17  23:37:34  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1995/03/12  17:09:29  sanjeev
 * changes for new msg macros
 *
 * Revision 1.2  1994/11/11  05:24:13  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:06  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef CHARE_H
#define CHARE_H

#include "converse.h"
#include "conv-mach.h"
#include "const.h"
#include "common.h"
#include "env_macros.h"

#ifndef NULL
#define NULL 0
#endif

#ifdef DEBUG
#define TRACE(p) p
#else
#define TRACE(p)
#endif

#define NO_PACK		0
#define UNPACKED	1
#define PACKED		2



/* ---------------- the message declarations follow  ----------------------*/


/* DYNAMIC_BOC_INIT */
typedef struct {
        int ref;
        int source;
        ChareNumType ep;
        ChareIDType id;
} DYNAMIC_BOC_REQUEST_MSG;

typedef struct {
	ChareNumType boc;
	int ref;
} DYNAMIC_BOC_NUM_MSG;


typedef struct bocinit_queue{
	void **block;
	short block_len;
	short first;
	short length;
} BOCINIT_QUEUE;
/**** end changes by Milind *****/

typedef struct dummymsg {
	int x;
} DummyMsg;


/* include macros for bit vector priorities */
#include "msg_macros.h"
#include "prio_macros.h"
#include "sys_macros.h"
#include "communication.h"

#endif
