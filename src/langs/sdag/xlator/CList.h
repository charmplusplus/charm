#ifndef _TList_H_
#define _TList_H_

#include <stdio.h>

// Quick and dirty List for small numbers of items.
// This should ideally be a template but for portability's sake, its
// a generic list of (void *)

class TList
{
  private:

    class Elem
    {
      public:
      Elem *next;
      void *data;
      Elem(void *d) : next(NULL), data(d) {;}
    };

    Elem *first;
    Elem *current;
    unsigned int len;

  public:

    TList(void) : first(0), len(0) {;}

    int empty(void) { return ! first; }
    
    void *begin(void) {
      current = first;
      if(current==(Elem *)0)
        return (void *) 0;
      else
        return current->data;
    }


    int end(void) {
      return (current == 0);
    }

    void *next (void) {
      current = current->next;
      if(current==(Elem *)0)
	return (void *)0;
      else
        return current->data;
    }

    void *front(void)
    {
      return ((first==0) ? 0 : first->data);
    }

    void *pop(void)
    {
      void *data;
      if( first!= 0) {
        data = first->data;
        Elem *nn = first->next;
        delete first;
        first = nn;
        len --;
      } else {
        data = 0;
      }
      return data;
    }

    void remove(void *data)
    {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first->data == data) {
        first = first->next;
        len --;
        return;
      }
      // case 3: middle or last element to be removed
      Elem *nn;
      Elem *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn->data == data) {
          prev->next = nn->next;
          len --;
          return;
        }
        prev = nn;
      }
    }

    void append(void *data)
    {
      if(first == 0) {
        first = new Elem(data);
      } else {
        Elem *nn;
        for( nn = first ; nn->next ; nn = nn->next );
        nn->next = new Elem(data);
      }
      len++;
    }

    int length(void) {
      return len;
    }
};

#endif

