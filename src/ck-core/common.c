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
 * Revision 1.3  1995/04/02  00:46:58  sanjeev
 * changes for separating Converse
 *
 * Revision 1.2  1995/03/12  17:09:12  sanjeev
 * changes for new msg macros
 *
 * Revision 1.1  1994/11/03  17:38:35  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "performance.h"


/**********************************************************************/
/* This is perhaps the most crucial function in the entire system.    
Everything that a message does depends on these. To avoid computing
various offsets again and again, they have been made into variables,
computed only once at initialization. Any changes made to the layout
of the message must be reflected here. */
/**********************************************************************/
InitializeMessageMacros()
{
/* The message format is as follows :
        -------------------------------------
        | env | ldb | pad | user | priority |
        -------------------------------------
*/


#define ENVELOPE_SIZE sizeof(ENVELOPE)

/* count everything except the padding, then add the padding if needed */ 
	HEADER_SIZE = ENVELOPE_SIZE + LDB_ELEM_SIZE ;

	if (HEADER_SIZE%8 == 0)
		PAD_SIZE = 0;
	else {
       		PAD_SIZE = 8 - (HEADER_SIZE%8);
	    	HEADER_SIZE += PAD_SIZE;
	}

	/********************* ENVELOPE **************************************/

	_CK_Env_To_Usr = ENVELOPE_SIZE + LDB_ELEM_SIZE + PAD_SIZE ;


	/******************** LDB ********************************************/

 	_CK_Ldb_To_Usr = LDB_ELEM_SIZE + PAD_SIZE ;


	/******************* USR *********************************************/

 	_CK_Usr_To_Env = -(ENVELOPE_SIZE + LDB_ELEM_SIZE + PAD_SIZE) ;
 	_CK_Usr_To_Ldb = -(LDB_ELEM_SIZE + PAD_SIZE);
}	
