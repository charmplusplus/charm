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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.7  1995/05/04  22:03:51  jyelon
 * *** empty log message ***
 *
 * Revision 1.6  1995/05/03  20:57:13  sanjeev
 * bug fixes for finding uninitialized modules
 *
 * Revision 1.5  1995/04/02  00:47:16  sanjeev
 * changes for separating Converse
 *
 * Revision 1.4  1995/03/12  17:08:13  sanjeev
 * changes for new msg macros
 *
 * Revision 1.3  1995/01/17  23:45:52  knauff
 * Added variables for the '++outputfile' option in the network version.
 *
 * Revision 1.2  1994/12/01  23:55:06  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:32  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "trans_defs.h"
#include <stdio.h>

/**********************************************************************/
/* These fields are needed by message macros. Any changes must be
reflected there. */
/**********************************************************************/

int PAD_SIZE, HEADER_SIZE ;
int _CK_Env_To_Usr;
int _CK_Ldb_To_Usr;
int _CK_Usr_To_Env, _CK_Usr_To_Ldb ;




/**********************************************************************/
/* Other global variables. */
/**********************************************************************/
int TotalEps;
int NumReadMsg =  0;
int MsgCount; 			/* for the initial, pre-loop phase.
				to count up all the messages
		 		being sent out to nodes 		*/
int InsideDataInit = 0;
int mainChare_magic_number;
struct chare_block * mainChareBlock;
struct chare_block * currentChareBlock;
int currentBocNum = (NumSysBoc - 1); /* was set to  -1 */
int MainDataSize;  		/* size of dataarea for main chare 	*/


int 	     * EpLanguageTable ;

/* void            ** _CK_9_ReadMsgTable;  no longer global */
FUNCTION_PTR *ROCopyFromBufferTable, *ROCopyToBufferTable ;

int          * EpIsImplicitTable;
FUNCTION_PTR * EpTable; 	/* actual table to be allocated dynamically
			     	depending on the number of entry-points */
int 	     * EpToMsgTable ;  /* Table mapping EPs to associated 	
				messages.				*/
int 	     * EpChareTypeTable ;  /* Table mapping EPs to chare type 
				      (CHARE or BOC) */ 	
MSG_STRUCT   * MsgToStructTable;/* Table mapping message to struct table*/
PSEUDO_STRUCT * PseudoTable;

int	     *EpChareTable ;
char	     ** EpNameTable ;

int  	     * ChareSizesTable;
FUNCTION_PTR * ChareFnTable ;
char 	     ** ChareNamesTable;

int msgs_processed, msgs_created;

int nodecharesCreated=0;
int nodeforCharesCreated=0;
int nodebocMsgsCreated=0;
int nodecharesProcessed = 0;
int nodebocMsgsProcessed = 0;
int nodeforCharesProcessed = 0;

/* FIFO_QUEUE *LocalQueueHead; now in converse.c */

int PrintChareStat = 0;
int PrintSummaryStat = 0;

int RecdStatMsg;
int RecdPerfMsg;

int numHeapEntries=0;	/* heap of tme-dep calls   */
int numCondChkArryElts=0; /* arry hldng conditon check info */


int _CK_13PackOffset;
int _CK_13PackMsgCount;
int _CK_13ChareEPCount;
int _CK_13TotalMsgCount;
FUNCTION_PTR    * _CK_9_GlobalFunctionTable;

int MainChareLanguage = -1 ;
