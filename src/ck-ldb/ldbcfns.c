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
 * Revision 2.1  1995-07-07 01:06:48  narain
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

CpvDeclare(int, LDB_ELEM_SIZE);

void ldbModuleInit()
{
  CpvInitialize(int, LDB_ELEM_SIZE);

  CpvAccess(LDB_ELEM_SIZE) = getLdbSize();
}

void CldPickFreeChare(void **a)
{
}

LdbAddSysBocEps()
{
}

CpvDeclare(int, Cldcurrent_seednum);
CpvDeclare(int, Cldnumseeds);

typedef struct Cldtokholder {
  int seed_num;
  void (*sendfn)();
  void *msgptr;
  struct Cldtokholder *next;
} CldTOK_HOLDER;

CpvDeclare(CldTOK_HOLDER*, Cldtokenlist);
CpvDeclare(CldTOK_HOLDER*, Cldlasttoken);
CpvDeclare(int, Cldhandlerid);

CldTOK_HOLDER *new_CldTOK_HOLDER(void (*sendfn)(), void *msgptr)
{
  CldTOK_HOLDER *ret = (CldTOK_HOLDER *)malloc(sizeof(CldTOK_HOLDER));
  ret->seed_num = CpvAccess(Cldcurrent_seednum)++;
  CpvAccess(Cldnumseeds)++;
  ret->sendfn = sendfn;
  ret->msgptr = msgptr;
  ret->next = NULL;
}

/******************************************************************
 * This function adds a token to the end of the token list 
 ******************************************************************/
int CldAddToken(void *msg, void (*sendfn))
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
  enqmsg = (void *)malloc(CmiMsgHeaderSizeBytes + sizeof(int));
  CmiSetHandler(enqmsg, CpvAccess(Cldcurrent_seednum));
  CsdEnqueue(enqmsg);
}

/*******************************************************************
 * Removes the token from the list of token's if it exists 
 *******************************************************************/
CldTOK_HOLDER *CldRemoveToken(int seednum)
{
  CldTOK_HOLDER *ret, *temp;
  if(CpvAccess(Cldtokenlist) == NULL)
    return NULL;
  if((CpvAccess(Cldtokenlist))->seed_num == seednum)
    {
      ret = CpvAccess(Cldtokenlist);
      CpvAccess(Cldtokenlist) = ret->next;

      if(CpvAccess(Cldtokenlist) == NULL) /* Empty token list */
	CpvAccess(Cldlasttoken) = NULL; 
    }
  else
    {
      temp = CpvAccess(Cldtokenlist);
      while(temp->next && temp->next->seed_num != seednum)
	temp = temp->next;
      if(temp->next)
	{
	  ret = temp->next;
	  temp->next = ret->next;

	  if(temp->next == NULL)
	    CpvAccess(Cldlasttoken) = temp; /* Update Cldlasttoken */
	}
    }
  if(ret != NULL)
    CpvAccess(Cldnumseeds)--;
  return ret;
}

/********************************************************************
 * Handler for the Load balancer, gets the token out of the list and
 * calls its handler 
 ********************************************************************/
Cldhandler(void *msg)
{
  CldTOK_HOLDER *tok;
  void *dispmsg;
  int seednum = (int)(*((char *)msg + CmiMsgHeaderSizeBytes));
  if((tok = CldRemoveToken(seednum)) != NULL)
    (CmiGetHandlerFunction(tok->msgptr))(tok->msgptr);
}


int Cldbtokensinit()
{
  CpvAccess(Cldtokenlist) = NULL;
  CpvAccess(Cldlasttoken) = NULL;
  CpvAccess(Cldnumseeds) = 0;
  CpvAccess(Cldcurrent_seednum) = 0;
  CpvAccess(Cldhandlerid) = CmiRegisterHandler(Cldhandler);
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

  if(CpvAccess(Cldnumseeds))
    {
      ret = CpvAccess(Cldtokenlist);
      CpvAccess(Cldtokenlist) = ret->next;
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
int CldPickSeedAndSend(int pe)
{
  CldTOK_HOLDER *temp = CldPickFreeSeed();
  if(temp != NULL)
    {
      (*(temp->sendfn))(temp->msgptr, pe);
      return 1;  
    }
  return 0;
}
