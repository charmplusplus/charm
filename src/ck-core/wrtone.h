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
 * Revision 2.0  1995-06-02 17:30:04  brunner
 * Reorganized directory structure
 *
 * Revision 1.3  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/11/11  05:24:46  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:15  brunner
 * Initial revision
 *
 ***************************************************************************/
/****************************************************************************

   Types and stuff for write-once variables

****************************************************************************/

#define MAXWRITEONCEVARS  50         /* arbitrary maximum number */

#define isLeaf(peNum) (CmiNumSpanTreeChildren(peNum) == 0)

/* Data Structures. We keep a different type for the host and for the nodes */
/* This is because on the nodes we don't need a lot of the information that */
/* is kept on the host to inform the user program that the wov has been     */
/* created.                                                                 */

typedef struct {              /* data needed for each write once variable */
    EntryPointType ep;
    ChareIDType    cid;
    int		   numAcks;
    int            wovSize;
    char           *wovData;
    } WOV_Elt_Data;

/* this is the struct that holds the local boc data */
typedef struct {           
    int numWOVs;              /* current number of write once variables    */
    WOV_Elt_Data WOVArray[MAXWRITEONCEVARS];
    } WOV_Boc_Data;

/* the rest of the structs are messaged that are passed around */
typedef struct {            /* Original message sent from node up To the host.*/
    ChareIDType    cid;
    EntryPointType ep;
    int            wovSize;
    } Host_New_WOV_Msg;

typedef struct {            /* Message sent from the host down to the nodes   */
    WriteOnceID    wovID;
    int            wovSize;
    } Node_New_WOV_Msg;

typedef struct {            /* Acknowledge msg passed up the tree to the host.*/
    WriteOnceID    wovID;
    } Ack_To_Host_Msg;

typedef struct {            /* Final message returned to the user program.    */
    WriteOnceID    wovID;
    } Return_To_Origin_Msg;

