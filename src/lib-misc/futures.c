/*this module provides simple sequential functions that (help) implement
the "Futures" abstraction.  Creating a new future simply returns an integer,
after creating an internal data structure.  Two additional functions are 
provided: one to set the value of the future, and another to wait for the
future to be available.  The latter (waiting) function should be called
only from a thread.  It returns with the value of the future (a void*)
if it is already available.  Otherwise, it suspends until the future is
available, an then returns with its value.  Multiple threads may wait for
the same future.  The future is freed year via an explicit call, or by
setting a parameter during the wait call. */

#include <converse.h>

#define  FREE  0
#define WAITING 1
#define AVAILABLE 2
#define NULL 0

typedef struct elt {
  /*struct elt */ void * next;
  CthThread t;
} Element;

typedef struct {
  int ready; /* 0: free; 1: waiting; 2: available */
  void *value;
  Element *waiters;
  int count;/* optional.  Reference count. */
} Future;
 
Future* futures;  /* Array of futures*/
  int size =  100;
/* change these to  csv later */

CsvDeclare(int, futuresHandler);

sendToFuture(void *m, int key)
{
*((int *) m ) =  CsvAccess(futuresHandler);
*((int *) ((char *)m + CmiMsgHeaderSizeBytes)) = key;
}


void setFutureHandler (void *m)
{
CmiGrabBuffer(m);
setFuture( *((int *) ((char *)m + CmiMsgHeaderSizeBytes)), m);
}


/* Futures abstraction */

futuresInitialize()
{ 
  int i;
  futures =  CmiAlloc( size*sizeof(Future ) );
  /*  register the handler */
  for (i=0; i<size; i++ )
    { futures[ i].ready = FREE;
    futures[ i].waiters = NULL;
    }
  CsvAccess(futuresHandler) = CmiRegisterHandler(setFutureHandler);
}

int createFuture()
{
int i;
if (!futures) futuresInitialize ();
for (i=0; i<size; i++ )
  if (futures[ i].ready == FREE) break;

if (i==size) { CmiPrintf(" future creation failed.\n"); return (-1); }
futures[ i].ready = WAITING;
futures[ i].waiters = NULL;
return(i);

}

int destroyFuture(int key)
     /* this function added on 7/22/97. -- sanjay */
{
  futures[ i].ready = FREE;
  futures[ i].waiters = NULL;
  CmiFree(futures[ i].value);
  futures[ i].value = NULL;

}

static void* reallyWait (int key)
  {
     Element* e = (Element *) CmiAlloc(sizeof (Element) );

     e->next =  futures[ key].waiters;
     e->t = CthSelf();
     futures[ key].waiters = e;
     CthSuspend();
     return( futures[ key].value);
  }

static void awaken_threads (Element *e)
  {  Element *f;
    while (e)
      {
/* CmiPrintf("[%d] Awakening next thread\n",CmiMyPe()); */
	CthAwaken(e->t);
	f = e;
	e = e->next;
	CmiFree( f);
      }
  }


setFuture(int key, void*pointer)
  {
    if (!futures) futuresInitialize ();
    futures[ key].ready = AVAILABLE;
    futures[ key].value = pointer;

    awaken_threads (futures[ key].waiters);
    CmiPrintf(" [%d] awakened them\n",CmiMyPe());
  }

void* waitFuture (int key, int free)
  {
    static void* reallyWait ();
    if (!futures) futuresInitialize ();

    if (futures[ key].ready == AVAILABLE) {
      if (free)   {
	 futures[ key].ready = FREE;
      }
      /*      if (--count == 0)  free the record.*/
      return(futures[ key].value);
    }
    else {
      /* insert thread in queue and Suspend thread */
       return (reallyWait (key));
    }
  }

