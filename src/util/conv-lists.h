#ifndef _CONV_LISTS_H
#define _CONV_LISTS_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
  @file 
  Definitions of CdsFifo routines.
  @ingroup ConverseScheduler

  @addtogroup ConverseScheduler
  @{
 */
typedef void *CdsFifo;

CdsFifo CdsFifo_Create(void);
CdsFifo CdsFifo_Create_len(int len);
void    CdsFifo_Enqueue(CdsFifo q, void *elt);
void *  CdsFifo_Dequeue(CdsFifo q);
void    CdsFifo_Push(CdsFifo q, void *elt);
void *  CdsFifo_Pop(CdsFifo q);
void    CdsFifo_Destroy(CdsFifo q);
void ** CdsFifo_Enumerate(CdsFifo q);
int     CdsFifo_Empty(CdsFifo q);
void *  CdsFifo_Peek(CdsFifo q);
int     CdsFifo_Length(CdsFifo q);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
