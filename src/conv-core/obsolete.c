
 This function gets all outstanding messages out of the network, executing
 their handlers if they are for this lang, else enqueing them in the FIFO 
 queue

int
CmiClearNetworkAndCallHandlers(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang )  {
        if (CpvAccess(CmiLastBuffer)) CmiFree(CpvAccess(CmiLastBuffer));
        CpvAccess(CmiLastBuffer) = msg;
        (CmiGetHandlerFunction(msg))(msg);
        retval = 1;
      } else {
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      }
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  while ( (msg = CmiGetNonLocal()) != NULL ) {
    if (CmiGetHandler(msg)==lang) {
      if (CpvAccess(CmiLastBuffer)) CmiFree(CpvAccess(CmiLastBuffer));
      CpvAccess(CmiLastBuffer) = msg;
      (CmiGetHandlerFunction(msg))(msg);
      retval = 1;
    } else {
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  }
  return retval;
}

 
 Same as above function except that it does not execute any handler functions

int
CmiClearNetwork(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	  retval = 1;
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  while ( (msg = CmiGetNonLocal()) != NULL ) {
    if (CmiGetHandler(msg)==lang) 
      retval = 1;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  return retval;
}


