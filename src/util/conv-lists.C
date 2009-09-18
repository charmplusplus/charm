#include "conv-lists.h"
#include "cklists.h"

/** 
 * @file 
 * Declarations of CdsFifo* routines
 *
 *  @ingroup ConverseScheduler
 *  @addtogroup ConverseScheduler
 *  @{
 */

typedef CkQ<void*> _Fifo;

CdsFifo CdsFifo_Create(void) { return (CdsFifo) new _Fifo(); }
CdsFifo CdsFifo_Create_len(int len) { return (CdsFifo) new _Fifo(len); }
void    CdsFifo_Enqueue(CdsFifo q, void *elt) { ((_Fifo*)q)->enq(elt); }
void *  CdsFifo_Dequeue(CdsFifo q) { return ((_Fifo*)q)->deq(); }
void    CdsFifo_Push(CdsFifo q, void *elt) { ((_Fifo*)q)->push(elt); }
void *  CdsFifo_Pop(CdsFifo q) { return ((_Fifo*)q)->deq(); }
void    CdsFifo_Destroy(CdsFifo q) { delete ((_Fifo*)q); }
void ** CdsFifo_Enumerate(CdsFifo q) { return ((_Fifo*)q)->getArray(); }
int     CdsFifo_Empty(CdsFifo q) { return (int)((_Fifo*)q)->isEmpty(); }
void *  CdsFifo_Peek(CdsFifo q) { 
  return ((_Fifo*)q)->length() ? (*((_Fifo*)q))[0] : 0; 
}
int     CdsFifo_Length(CdsFifo q) { return ((_Fifo*)q)->length(); }


/** 
 *   @}
 */
