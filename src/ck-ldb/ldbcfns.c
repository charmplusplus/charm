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
 * Revision 2.7  1997-10-29 23:52:57  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.6  1997/07/30 17:31:04  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/11/06 17:55:09  milind
 * Changed CldAdd Token to take priority info as parameters.
 *
 * Revision 2.4  1995/10/13  18:15:22  jyelon
 * K&R changes, etc.
 *
 * Revision 2.3  1995/07/19  22:15:21  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/07/09  17:50:46  narain
 * Made changes in tokens registered with Scheduler.. debugged code.
 *
 * Revision 2.1  1995/07/07  01:06:48  narain
 * *** empty log message ***
 *
 * Revision 2.0  1995/07/07  00:43:58  narain
 * *** empty log message ***
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "converse.h"

#ifndef NULL
#define NULL 0
#endif

void Cldhandler(void *);

typedef struct Cldtokholder {
  char msg_header[CmiMsgHeaderSizeBytes];
  void (*sendfn)();
  void *msgptr;
  int sent_out;
  struct Cldtokholder *next;
} CldTOK_HOLDER;

CpvDeclare(int, LDB_ELEM_SIZE);
CpvDeclare(int, Cldnumseeds);
CpvDeclare(CldTOK_HOLDER*, Cldtokenlist);
CpvDeclare(CldTOK_HOLDER*, Cldlasttoken);
CpvDeclare(int, Cldhandlerid);

void CldModuleInit()
{
  CpvInitialize(int, LDB_ELEM_SIZE);
  CpvInitialize(int, Cldnumseeds);
  CpvInitialize(CldTOK_HOLDER*, Cldtokenlist);
  CpvInitialize(CldTOK_HOLDER*, Cldlasttoken);
  CpvInitialize(int, Cldhandlerid);

  CpvAccess(LDB_ELEM_SIZE) = CldGetLdbSize();
  CpvAccess(Cldhandlerid) = CmiRegisterHandler(Cldhandler);
  CpvAccess(Cldtokenlist) = NULL;
  CpvAccess(Cldlasttoken) = NULL;
  CpvAccess(Cldnumseeds) = 0;
}

void CldPickFreeChare(a)
    void **a;
{
}

CldTOK_HOLDER *new_CldTOK_HOLDER(sendfn, msgptr)
    void (*sendfn)(); void *msgptr;
{
  CldTOK_HOLDER *ret = (CldTOK_HOLDER *)CmiAlloc(sizeof(CldTOK_HOLDER));
  CpvAccess(Cldnumseeds)++;
  ret->sendfn = sendfn;
  ret->msgptr = msgptr;
  ret->sent_out = 0;
  ret->next = NULL;

  return ret;
}

/******************************************************************
 * This function adds a token to the end of the token list 
 ******************************************************************/
void CldAddToken(msg, sendfn, queuing, priolen, prioptr)
    void *msg; void (*sendfn)();
    unsigned int queuing, priolen, *prioptr;
{
  void *enqmsg;
  CldTOK_HOLDER *newtok = new_CldTOK_HOLDER(sendfn, msg), *temp;

  /* Add a token to the list of tokens */
  temp = CpvAccess(Cldlasttoken);
  if(temp)
    {
      temp->next = newtok;
      CpvAccess(Cldlasttoken) = newtok;
    }
  else /* Empty list of tokens */
    CpvAccess(Cldtokenlist) = CpvAccess(Cldlasttoken) = newtok;

  /* Add a token in the scheduler's queue to call this strategy */

  enqmsg = (void *)newtok;
  CmiSetHandler(enqmsg, CpvAccess(Cldhandlerid));
  CsdEnqueue(enqmsg);
}

/*******************************************************************
 * Removes the token from the list of token's if it exists 
 *******************************************************************/
int CldRemoveToken(tok)
    CldTOK_HOLDER *tok;
{
  CldTOK_HOLDER *temp;

  if(CpvAccess(Cldtokenlist) == NULL)
    return 0;
  if((CpvAccess(Cldtokenlist)) == tok)
    {
      CpvAccess(Cldtokenlist) = tok->next;

      if(CpvAccess(Cldtokenlist) == NULL) /* Empty token list */
	CpvAccess(Cldlasttoken) = NULL; 
    }
  else
    {
      temp = CpvAccess(Cldtokenlist);
      while(temp->next && temp->next != tok)
	temp = temp->next;
      if(temp->next)
	{
	  temp->next = tok->next;

	  if(temp->next == NULL)
	    CpvAccess(Cldlasttoken) = temp; /* Update Cldlasttoken */
	}
    }
  if(tok->sent_out == 0) /* The number of seeds is decreased in the 
			    CldPickFreeSeed call */
    CpvAccess(Cldnumseeds)--;
  return ((tok->sent_out)?0:1);
}

/********************************************************************
 * Handler for the Load balancer, gets the token out of the list and
 * calls its handler 
 ********************************************************************/
void Cldhandler(msg)
    void *msg;
{
  CldTOK_HOLDER *tok;
  tok = (CldTOK_HOLDER *)msg;
  if(CldRemoveToken(tok))
    (CmiGetHandlerFunction(tok->msgptr))(tok->msgptr);
}


/*********************************************************************
 * Load function : Returns the number of seeds in the token list 
 *********************************************************************/
int CldMyLoad()
{
  return CpvAccess(Cldnumseeds);
}

/*******************************************
 * Pick a free seed for redistribution
 * Return NULL if no seed is available
 *******************************************/
CldTOK_HOLDER *CldPickFreeSeed()
{
  CldTOK_HOLDER *ret = NULL;

  ret = CpvAccess(Cldtokenlist);
  while(ret && ret->sent_out)
    ret = ret->next;
  
  if(ret)
    {
      ret->sent_out = 1;
      CpvAccess(Cldnumseeds)--;
      if(CpvAccess(Cldnumseeds) == 0)
	CpvAccess(Cldlasttoken) = NULL;
    }
  return ret;
}

/******************************************************
 * Pick a free seed and send to processor 'pe'
 *  - Return value of 1 indicates a seed was sent out
 *  -              of 0 indicates no seed was sent out
 ******************************************************/
int CldPickSeedAndSend(pe)
    int pe;
{
  CldTOK_HOLDER *temp = CldPickFreeSeed();
  if(temp != NULL)
    {
      (*(temp->sendfn))(temp->msgptr, pe);
      return 1;  
    }
  return 0;
}


