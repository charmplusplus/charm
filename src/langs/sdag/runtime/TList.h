#ifndef _TList_H_
#define _TList_H_

#include <stdio.h>
#include "CWhenTrigger.h"
#include "CMsgBuffer.h"

// Quick and dirty List for small numbers of items.
// It should ideally be a template, but in order to have portability,
// we would make it two lists

class TListCWhenTrigger
{
  private:

    CWhenTrigger *first, *last;
    CWhenTrigger *current;

  public:

    TListCWhenTrigger(void) : first(0), last(0) {;}

    int empty(void) { return ! first; }
    
    CWhenTrigger *begin(void) {
      return (current = first);
    }

    int end(void) {
      return (current == 0);
    }

    CWhenTrigger *next (void) {
      return (current = current->next);
    }

    CWhenTrigger *front(void)
    {
      return first;
    }

    void remove(CWhenTrigger *data)
    {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first == data) {
        first = first->next;
	if(first==0) last=0;
        return;
      }
      // case 3: middle or last element to be removed
      CWhenTrigger *nn;
      CWhenTrigger *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn == data) {
          prev->next = nn->next;
	  if(nn==last)
	    last=prev;
          return;
        }
        prev = nn;
      }
    }

    void append(CWhenTrigger *data)
    {
      data->next = 0;
      if(first == 0) {
        last = first = data;
      } else {
        last->next = data;
	last = last->next;
      }
    }
};

class TListCMsgBuffer
{
  private:

    CMsgBuffer *first, *last;
    CMsgBuffer *current;

  public:

    TListCMsgBuffer(void) : first(0), last(0) {;}

    int empty(void) { return ! first; }
    
    CMsgBuffer *begin(void) {
      return (current = first);
    }

    int end(void) {
      return (current == 0);
    }

    CMsgBuffer *next (void) {
      return (current = current->next);
    }

    CMsgBuffer *front(void)
    {
      return first;
    }

    void remove(CMsgBuffer *data)
    {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first == data) {
        first = first->next;
	if(first==0) last=0;
        return;
      }
      // case 3: middle or last element to be removed
      CMsgBuffer *nn;
      CMsgBuffer *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn == data) {
          prev->next = nn->next;
	  if(nn==last)
	    last=prev;
          return;
        }
        prev = nn;
      }
    }

    void append(CMsgBuffer *data)
    {
      data->next = 0;
      if(first == 0) {
        last = first = data;
      } else {
        last->next = data;
	last = last->next;
      }
    }
};

#endif

