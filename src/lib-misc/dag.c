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
 * Revision 1.5  1998-02-27 11:53:02  jyelon
 * Cleaned up header files, replaced load-balancer.
 *
 * Revision 1.4  1997/08/18 18:02:03  milind
 * Located and fixed a bug reported by Ed. dag.c was using CmiFree to free
 * charm messages instead of CkFreeMsg.
 *
 * Revision 1.3  1997/03/25 15:04:56  milind
 * Made changes suggested by Ed Kornkven to fix bugs in Dagger.
 *
 * Revision 1.2  1996/03/09 18:11:20  jyelon
 * Fixed Ck --> Cmi
 *
 * Revision 1.1  1995/06/13 11:32:16  jyelon
 * Initial revision
 *
 * Revision 1.2  1995/04/14  05:26:11  milind
 * changed redefinistions of TRUE and FALSE
 *
 * Revision 1.1  1994/11/03  17:35:21  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include "charm.h"

/* _dag3_WLIMIT is defien d in dag.c dagger..h also */
#define _dag3_WLIMIT 8
#define _dag3_FREEBLIMIT 4
#define _dag3_FREECLIMIT 4
#define _dag3_FREERLIMIT 4

struct s_dag3_DAGVAR {
    int index;
    int init_value;
    int counter;
};

struct s_dag3_RLNODE {
    int    wno;
    int    refnum;
    struct s_dag3_RLNODE *next;
};

struct s_dag3_BUFFER {
    int eno;
    int refnum;
    int expect;
    int ecount;
    int free_count;
    void *msg;
    struct s_dag3_BUFFER **prev;
    struct s_dag3_BUFFER *next;
};


struct s_dag3_COUNT {
    int refnum;
    int value;
    int bix;
    struct s_dag3_BUFFER *bpa[_dag3_WLIMIT];
    struct s_dag3_COUNT  **prev;
    struct s_dag3_COUNT  *next;
};

struct s_dag3_FREELIST {
    int bcount;
    int ccount;
    struct s_dag3_BUFFER *b;
    struct s_dag3_COUNT  *c;
};

struct s_dag3_RL {
    int    dagexit;
    struct s_dag3_COUNT *head;
    struct s_dag3_COUNT *tail;
};

typedef struct s_dag3_COUNT _dag3_RLNODE;
typedef struct s_dag3_RL     _dag3_RL;
typedef struct s_dag3_BUFFER _dag3_BUFFER;
typedef struct s_dag3_COUNT  _dag3_COUNT;
typedef struct s_dag3_FREELIST _dag3_FREELIST;
typedef struct s_dag3_DAGVAR _dag3_DAGVAR;
typedef struct s_dag3_DAGVAR DAGVAR;


_dag3_BUFFER *_dag4_allocb();
_dag3_COUNT  *_dag4_allocc();


#define _DAG4_ALLOCC(l) _dag4_allocc(l)
#define _DAG4_ALLOCB(l) _dag4_allocb(l)

#define _DAG4_FREEC(l,p) _dag4_freec(l,p)
#define _DAG4_FREEB(l,p) _dag4_freeb(l,p)

/*
#define _DAG4_ALLOCC(l) (_dag3_COUNT *)CmiAlloc(sizeof(_dag3_COUNT))
#define _DAG4_ALLOCB(l) (_dag3_BUFFER *)CmiAlloc(sizeof(_dag3_BUFFER))

#define _DAG4_FREEC(l,p) CmiFree(p)
#define _DAG4_FREEB(l,p) CmiFree(p)
*/

#define MATCH    1
#define MULTIPLE 2
#define ANY      4
#define AVAILABLE ((void *) NULL)
#define PROCESSED ((void *) 1)

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* matching_find_count */
/* ****************************************************************** */
/* search for the counter node for the when block wno with the        */
/* refnumber refnum. If there is no counter node, create it.          */
/* The counter nodes are kept in a linked list sorted by refnum       */
/* Output: pointer to the counter node                                */
/* ****************************************************************** */

_dag3_COUNT *_dag4_mfc(flist,count,counter,wno,refnum)
_dag3_FREELIST *flist;
int             wno,refnum,count[]; 
_dag3_COUNT     *counter[];
{   int         flag;
    _dag3_COUNT *prev,*current,*count_node;

    prev    = (_dag3_COUNT *) NULL;
    current = counter[wno];              /* list head */
    flag = 3;
    if (current == (_dag3_COUNT *) NULL) 
        flag = 1;                        /* empty list */
    else 
        while(current != (_dag3_COUNT *) NULL ) {
          if (current->refnum == refnum) {flag=0;break;}  /* found */
          if (current->refnum > refnum)  {flag=2;break;}  /* not found */
          prev = current; 
          current = current->next; 
        }

    if (flag) {
        /* Create a counter node */
        count_node = _DAG4_ALLOCC(flist);
        count_node->refnum = refnum;
        count_node->value  = count[wno];
        count_node->bix    = 0;
        count_node->next = (_dag3_COUNT *) NULL;
    }


    /* put in  the list in a sorted order */
    switch (flag) {
       case 0: return current;
       case 1: counter[wno] = count_node; 
               count_node->prev = &(counter[wno]);
               break;
       case 2: if (prev == (_dag3_COUNT *) NULL ) {
                  counter[wno] = count_node;
                  count_node->prev = &(counter[wno]);
                }
               else {
                  prev->next = count_node;
                  count_node->prev = &(prev->next);
               }
               count_node->next = current;
               current->prev = &(count_node->next);
               break;
       case 3: prev->next = count_node;
               count_node->prev = &(prev->next);
   
    }

    return(count_node);
}







/* nonmatching_find_count */
/* ****************************************************************** */
/* return the counter node for the when block wno                     */ 
/* If there is no counter node, create it.                            */
/* Since reference number is ignored, there exist one counter node    */
/* Output: pointer to the counter node                                */
/* ****************************************************************** */

_dag3_COUNT *_dag4_nonmfc(flist,count,counter,wno)
_dag3_FREELIST *flist;
int            count[],wno;
_dag3_COUNT    *counter[];
{
    _dag3_COUNT *count_node;

    if (counter[wno] == (_dag3_COUNT *) NULL) {
       count_node        = _DAG4_ALLOCC(flist);
       count_node->refnum= 0;
       count_node->value = count[wno]; 
       count_node->bix   = 0;
       count_node->next  = NULL;
       count_node->prev  = &(counter[wno]);
       counter[wno]      = count_node;
    }
    return (counter[wno]);
}






/* matching_find_buffer */
/* *****************************************************************  */
/* search for the message buffer for the entry point eno, with the    */
/* reference number refnum. If there is no buffer, create an empty    */
/* one. Th ebuffer list ismanaged as a sorted linked list by refnum.  */
/* output : pointer to the buffer node.                               */
/* *****************************************************************  */

_dag3_BUFFER * _dag4_mfb(flist,ep_buffer,eno,refnum)
_dag3_FREELIST *flist;
_dag3_BUFFER *ep_buffer[];
int          eno,refnum;
{
   int          flag;
   _dag3_BUFFER *prev,*current,*buffer_node;

   prev   = (_dag3_BUFFER *) NULL;
   current = ep_buffer[eno];                    /* list head */
   flag = 3;

   if (current == (_dag3_BUFFER *) NULL ) 
      flag = 1;                                 /* empty list */
   else
      while (current) {
        if (current->refnum == refnum) {flag=0; break;} /* found */
        if (current->refnum > refnum)  {flag=2; break;} /* not found */
        prev = current;
        current = current->next;
      }

   if (flag) {
      buffer_node          = _DAG4_ALLOCB(flist);
      buffer_node->eno     = eno;
      buffer_node->next    = (_dag3_BUFFER *) NULL;
      buffer_node->msg     = (void *) NULL;
      buffer_node->expect  = FALSE;
      buffer_node->ecount  = -1;
      buffer_node->refnum  = refnum;
   }

   switch (flag) {
     case 0: return current;
     case 1: ep_buffer[eno] = buffer_node;
             buffer_node->prev = &(ep_buffer[eno]); 
             break;
     case 2: if (prev == (_dag3_BUFFER *) NULL)  {
                 ep_buffer[eno] = buffer_node;
                 buffer_node->prev = &(ep_buffer[eno]);
                }
             else { 
                 prev->next = buffer_node;
                 buffer_node->prev = &(prev->next);
             }
             buffer_node->next = current; 
             current->prev     = &(buffer_node->next);
             break;
     case 3: prev->next = buffer_node;
             buffer_node->prev = &(prev->next); 
   }
   return (buffer_node);
}





/* nonmatching_find_buffer */
/* ****************************************************************  */
/* return the buffer node for the entry point eno                    */
/* There is only one buffer since the reference number is ignored    */
/* ****************************************************************  */

_dag3_BUFFER *_dag4_nonmfb(flist,ep_buffer,eno)
_dag3_FREELIST *flist;
_dag3_BUFFER *ep_buffer[];
int          eno;
{
   if (ep_buffer[eno] == (_dag3_BUFFER *) NULL) {
      ep_buffer[eno]       = _DAG4_ALLOCB(flist);
      ep_buffer[eno]->eno  = eno;
      ep_buffer[eno]->next = (_dag3_BUFFER *) NULL;
      ep_buffer[eno]->prev = &(ep_buffer[eno]);
      ep_buffer[eno]->msg  = (void *) NULL;
      ep_buffer[eno]->expect = FALSE;
      ep_buffer[eno]->ecount = -1; 
   }
   return ep_buffer[eno];
}





/* **************************************************************** */
/* Find buffer for the entry point eno.                             */
/* Output : ecount   : number of messages already arrived for a     */
/*                     multi-messag eentry point                    */
/*          msgcount : number of messages (for a multi-message      */
/*                     entry ) thar are not processed by a when-any */
/*                     block.                                       */
/* **************************************************************** */

_dag3_BUFFER * _dag4_fb(flist,ep_buffer,eno,etype,refnum,msgcount)
_dag3_FREELIST *flist;
int            eno,etype,refnum,*msgcount;
_dag3_BUFFER   *ep_buffer[];
{
   _dag3_BUFFER *buffer;

   if (etype & MATCH)
      buffer = _dag4_mfb(flist,ep_buffer,eno,refnum);
   else
      buffer = _dag4_nonmfb(flist,ep_buffer,eno);

   if (buffer->expect) CmiPrintf("dag error: multiple expect\n");
   buffer->expect = 1;

   *msgcount = 0;
   if (etype & MULTIPLE) {
      void **msgarray;
      msgarray  = (void **) buffer->msg;
      if (msgarray) 
         *msgcount = (int) msgarray[ ((int)msgarray[0])+1] - 1;
   }
   return buffer;
}




/* multi_put */
/* **************************************************************** */
/* put the message in the message array. This function is for       */
/* multi-message entry points.                                      */
/* **************************************************************** */ 
_dag4_mpm(buffer,msg,n) 
_dag3_BUFFER *buffer; 
void         *msg; 
int          n; 
{ 
   void **msgarray; 
   int  i,*index; 

   if (buffer->msg == (void *) NULL) { 
      msgarray      = (void **) CmiAlloc(sizeof(void *)*(2*n+2)); 
      msgarray[0]   = (void *) n; 
      msgarray[n+1] = (void *) 1;  /* index , initially == 1*/
      buffer->msg   = (void *) msgarray;
      for(i=1;i<=n;i++)  msgarray[i] = msgarray[i+n+1] = (void *) NULL;
     } 
   else 
     msgarray = (void **) buffer->msg; 
   
   index = (int *) (msgarray+n+1); 
   msgarray[(*index)++] = msg; 
   buffer->ecount--; 
}




/* ordinary_put */
/* **************************************************************** */
/* put the message in the buffer. This function is for              */
/* non multi-message entry points                                   */
/* **************************************************************** */

_dag4_opm(buffer,msg) 
_dag3_BUFFER *buffer; 
void         *msg;  
{ 
   if (buffer->msg == (void *) NULL){
        buffer->msg = msg; 
        buffer->ecount--; 
      } 
   else 
      CmiPrintf("dag error: unexpected message\n");
}





/* get_buffer */
/* ************************************************************** */
/* return a pointer to the message (or message array) in the      */
/* buffer list bl.                                                */
/* ************************************************************** */
void *_dag4_gb(rlnode,position,eno,etype)
_dag3_RLNODE *rlnode;
int          position,eno,etype;
{
    _dag3_BUFFER *buffer;
    void **msgarray;

   
    
/*
    for(i=0; i<rlnode->bix; i++) if (rlnode->bpa[i]->eno == eno) break;
*/
/* error message, remove it later */
    if (position >= rlnode->bix) printf("error in gb\n"); 

    buffer = rlnode->bpa[position];

    msgarray = (void **) buffer->msg;

    if ( !(etype & MULTIPLE)) 
        {
           buffer->free_count--;
           return buffer->msg;
        }
    else if (etype & ANY) {
            int i,j,n;
           
            n = (int) (msgarray[0]);
            for(i=n+2,j=1; j<=n; i++,j++)
               if (msgarray[i] == AVAILABLE ) {
                  msgarray[i] = PROCESSED;
                  return msgarray[j];
               }
            CmiPrintf("dag error: gb:can't find message\n"); 
         }
    else {
         buffer->free_count--;
         return (void *) &(msgarray[1]);
    }
}



/* **************************************************************** */
/* dispacth the when block wno for execution. Modify the ready list */
/* rl. If the when-block is of type ANY, then insert at the         */
/* beginning (i.e., higher priority)                                */
/* **************************************************************** */

_dag4_update_rl(flist,rl,wno,wtype,refnum,rlnode)
_dag3_FREELIST *flist;
_dag3_RL *rl;
int wno,wtype,refnum;
_dag3_RLNODE *rlnode;
{

    /* remove it from counter list */
    /* if it is of type ANY , then do nothing */
    if ( rlnode->prev) {
         *(rlnode->prev) = rlnode->next;
         if (rlnode->next) rlnode->next->prev = rlnode->prev;
    } 
  
    rlnode->next = NULL;
    rlnode->prev = NULL;

    if (rl->head == (_dag3_RLNODE *) NULL)
            rl->head = rl->tail = rlnode;
    else if (wtype & ANY ) {
            rlnode->next = rl->head;
            rl->head     = rlnode;
         }
    else {
            rl->tail->next = rlnode;
            rl->tail       = rlnode;
    }

    rlnode->value    = wno;
    rlnode->refnum = refnum;
}


/* *************************************************************** */
/* call the when-blocks which are eligible (the ones in rl).       */
/* until the ready list rl becomes empty.                          */
/* *************************************************************** */ 

_dag4_process_rl(flist,cklocalptr,rl,wsf,activator)
_dag3_FREELIST *flist;
void *cklocalptr;
_dag3_RL *rl;
int (*wsf)();
int *activator;
{
     int wno,refnum; 
     _dag3_RLNODE *rlnode;

     if (rl->dagexit) { /* if DagChare is issued, return */
        ChareExit();
        return;
     }
 
     *activator = 1; 

     while (  rl->head  ) {

        wno      = rl->head->value; 
        refnum   = rl->head->refnum;
        rlnode   = rl->head;
        rl->head = rlnode->next;
        rlnode->next = (_dag3_RLNODE *) NULL;  
        (*wsf)(cklocalptr,rlnode);
        _DAG4_FREEC(flist,rlnode); 
 
        /* Check if DagExit is executed, if so, return */
        if (rl->dagexit) {
           ChareExit();
           return;
        }


     }

     *activator = 0;
}




_dag4_freebuffer(flist,rlnode)
_dag3_FREELIST *flist;
_dag3_RLNODE   *rlnode;
{
     int          i,j,n;
     void         **msgarray;
     _dag3_BUFFER *buffer;

     for (i=0; i<rlnode->bix; i++) 
         if ( buffer = rlnode->bpa[i] )
            if (buffer->free_count == 0) {
               if (buffer->ecount == MULTIPLE) {
                  msgarray = (void **) buffer->msg;
                  if (msgarray) {
                    n = (int) (msgarray[0]);
                    for(j=1; j<=n; j++) if (msgarray[j]) CkFreeMsg(msgarray[j]);
                    CmiFree(msgarray);
                   }
                 }
               else {
                  if (buffer->msg) CkFreeMsg(buffer->msg);
               }

               *(buffer->prev) = buffer->next;
               if (buffer->next) buffer->next->prev = buffer->prev;
               _DAG4_FREEB(flist,buffer);
            }
}


/*
_dag4_m_freebuffer(flist,rlnode)
_dag3_FREELIST *flist;
_dag3_RLNODE   *rlnode;
{
     int  i,j,n;
     void **msgarray;
     _dag3_BUFFER *buffer;
     
     for(j=0; j<rlnode->bix; j++) 
        if ( buffer = rlnode->bpa[j] ) 
           if (buffer->free_count == 0) {
               msgarray = (void **) buffer->msg;
               if (msgarray) {
                   n = (int) (msgarray[0]);
                   for(i=1; i<=n; i++) if (msgarray[i]) CmiFree(msgarray[i]);
                   CmiFree(msgarray);
               }
               *(buffer->prev) = buffer->next;
               if (buffer->next) buffer->next->prev = buffer->prev;
               _DAG4_FREEB(flist,buffer);
           }
}
*/



_dag4_epconv(ep,table,n)
int ep,table[],n;
{
    int i;
    for(i=0; i<n; i++) if (table[i] == ep) return i;
}


_dag4_ccn(flist,position,wcount,wcounter,wno,wtype,rl,refnum,buffer)
_dag3_FREELIST *flist;
int            position,wcount[],wno,wtype,refnum;
_dag3_COUNT    *wcounter[];
_dag3_RL       *rl;
_dag3_BUFFER   *buffer;
{
    _dag3_COUNT *counter;

    counter = _dag4_nonmfc(flist,wcount,wcounter,wno);
    counter->bix++;
    counter->bpa[position] = buffer;
    if ( --(counter->value) == 0) {
          _dag4_update_rl(flist,rl,wno,wtype,refnum,counter);
          return TRUE;
    }
    return FALSE;
}


_dag4_ccm(flist,position,wcount,wcounter,wno,wtype,rl,refnum,buffer)
_dag3_FREELIST *flist;
int            position,wcount[],wno,wtype,refnum;
_dag3_COUNT    *wcounter[];
_dag3_RL       *rl;
_dag3_BUFFER    *buffer;
{
    _dag3_COUNT *counter;

    counter = _dag4_mfc(flist,wcount,wcounter,wno,refnum);
    counter->bix++;
    counter->bpa[position] = buffer;
    if ( --(counter->value) == 0) {
          _dag4_update_rl(flist,rl,wno,wtype,refnum,counter);
          return TRUE;
    }
    return FALSE;
}

_dag4_cci(flist,wno,wtype,rl,msgcount,refnum,buffer)
_dag3_FREELIST *flist;
int            wno,wtype,msgcount,refnum;
_dag3_RL       *rl;
_dag3_BUFFER   *buffer;
{
    int i;
    _dag3_COUNT *counter;

    for(i=0; i<msgcount; i++) {
       counter        = _DAG4_ALLOCC(flist);
       counter->value = wno;
       counter->refnum= refnum;
       counter->bix   = 1;
       counter->next  = NULL;
       counter->prev  = NULL;
       counter->bpa[0]= buffer;
       _dag4_update_rl(flist,rl,wno,wtype,refnum,counter);
    }
    return (msgcount > 0) ? TRUE : FALSE;
}


_dag3_BUFFER *_dag4_allocb(flist)
_dag3_FREELIST *flist;
{
      _dag3_BUFFER *temp;
      if (flist->b == NULL) 
         return (_dag3_BUFFER *) CmiAlloc(sizeof(_dag3_BUFFER));
      temp = flist->b;
      flist->b = flist->b->next;
      flist->bcount--;
      return temp;
}


_dag3_COUNT *_dag4_allocc(flist)
_dag3_FREELIST *flist;
{
      _dag3_COUNT *temp;
      if (flist->c == NULL) 
          return (_dag3_COUNT *) CmiAlloc(sizeof(_dag3_COUNT));
      temp = flist->c;
      flist->c = flist->c->next;
      flist->ccount--;
      return temp;
}

_dag4_freeb(flist,p)
_dag3_FREELIST *flist;
_dag3_BUFFER   *p;
{
      if (flist->bcount < _dag3_FREEBLIMIT) {
           p->next = flist->b;
           flist->b = p;
           flist->bcount++;
        }
      else CmiFree(p);
}
          

_dag4_freec(flist,p)
_dag3_FREELIST *flist;
_dag3_COUNT   *p;
{
      if (flist->ccount < _dag3_FREECLIMIT) {
           p->next = flist->c;
           flist->c = p;
           flist->ccount++;
        }
      else CmiFree(p);
}
