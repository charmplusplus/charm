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
 * Revision 2.1  1995-06-08 17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.2  1994/11/11  05:25:13  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:23  brunner
 * Initial revision
 *
 ***************************************************************************/
/***************************************************************************
 condsend.h

 Author: Wayne Fenton
 Date:   1/20/90
 Here are the data types used in conditionsends.c There are two basic types
 that are used in three major data structures. The two basic types are
 ChareDataEntry and BocDataEntry. One is used to send a msg and the other 
 is used to call a boc function. The three major data structures are

 1) a heap holding time values and ptrs to the basic types. The lowest times
    in the heap are checked against the current time. If lower, the msg is
    sent (boc func called).

 2) a stack holding ptrs to the basic types. Periodically, the stack is 
    traversed, and if we have a chareDataEntry then the function_ptr func
    is called. If it returns 1, then the msg is sent and the entry is 
    removed from the stack. If it is a BocDataEntry, then the boc func is
    called. If it returns 1, then the entry is removed from the stack.

 3) an array, each element of which corresponds to a known condition (ie
    queueEmpty). A linked list is attached to each condition, corresponding
    to actions wanted done when the condition occurs. When this happens (ie
    someone makes a call to RaiseCondition(CondNum), then a flag in the array
    is set. Then the next time through the PeriodicChecks routine, it will
    notice that the condition has been raised and execute the elements on
    the linked list (ie send a message or call a boc function).

***************************************************************************/
#ifndef CONDSEND_H
#define CONDSEND_H

#define ITSABOC                   0
#define ITSACHARE                 1

#define MAXTIMERHEAPENTRIES       512
#define MAXCONDCHKARRAYELTS       512

#define MAXIFCONDARISESARRAYELTS  512        /* just a dummy, no elts yet */
             /* the actual indices of the conditions should be defined here */

#define NUMSYSCONDARISEELTS       1    /* number of elements used by system */

#define QUEUEEMPTYCOND            0    /* queue empty condition */

typedef struct {
    FUNCTION_PTR   cond_fn;
    int            entry;
    void           *msg;
    int            size;
    ChareIDType    chareID;
    } ChareDataEntry;

typedef struct {
    int             bocNum;
    FUNCTION_PTR    fn_ptr;
    } BocDataEntry;             /* If pointed to from heap, then next is NULL */

typedef struct {
    unsigned int timeVal;     /* the actual time value we sort on           */
    int            bocOrChare;  /* so we know what kind of data we're ptng to */
    void           *theData;    /* points to either ChareDataEntry or Boc..   */
    } HeapIndexType;

typedef struct {
    int            bocOrChare;    /* same struct as HeapIndexType except that */
    void           *theData;      /* here we are using a simple stack to keep */
    } CondArrayEltType;           /* the information.                         */

typedef struct linkptr {
    int             bocOrChare;
    void            *theData;
    struct linkptr  *next;
    } LinkRec;

typedef struct {
    short          isCondRaised;  /* has condition been raised?               */
    LinkRec        *dataListPtr;  /* We keep a linked list for each element in*/
    } IfCondArisesArrayEltType;   /* the array. Each element in the list is   */
                                  /* either ChareDataEntry or BocDataEntry    */
#endif
