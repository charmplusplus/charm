#ifndef _TList_H_
#define _TList_H_

namespace xi {

// Quick and dirty List for small numbers of items.

template <class T>
class TList
{
  private:

    class Elem
    {
      public:
      Elem *next;
      T data;
      Elem(T d) : next(0), data(d) {;}
    };

    Elem *first;
    Elem *current;
    unsigned int len;

  public:

    TList(void) : first(0), len(0) {;}

    int empty(void) { return ! first; }
    
    T begin(void) {
      current = first;
      return (current ? current->data : T(0));
    }

    int end(void) {
      return (current == 0);
    }

    T next (void) {
      current = current->next;
      return (current ? current->data : T(0));
    }

    T front(void) {
      return ((first==0) ? 0 : first->data);
    }

    T pop(void) {
      T data = T(0);
      if(first) {
        data = first->data;
        Elem *nn = first->next;
        delete first;
        first = nn;
        len --;
      }
      return data;
    }

    void remove(T data) {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first->data == data) {
	Elem *tbr = first;
        first = first->next;
        len --;
	delete tbr;
        return;
      }
      // case 3: middle or last element to be removed
      Elem *nn;
      Elem *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn->data == data) {
          prev->next = nn->next;
          len --;
	  delete nn;
          return;
        }
        prev = nn;
      }
    }

    void append(T data) {
      Elem *e = new Elem(data);
      if(first == 0) {
        first = e;
      } else {
        Elem *nn;
        for( nn = first ; nn->next ; nn = nn->next );
        nn->next = e;
      }
      len++;
    }

    int length(void) {
      return len;
    }
};

}

#endif
