#ifndef _BLUE_TYPES_H_
#define _BLUE_TYPES_H_

/*****************************************************************************
   used internally, define minHeap of messages
   it use the msg time as key and dequeue the msg with the smallest time.
*****************************************************************************/

class minMsgHeap
{
private:
  char **h;
  int count;
  int size;
  void swap(int i, int j) {
    char * temp = h[i];
    h[i] = h[j];
    h[j] = temp;
  }
  
public:
  minMsgHeap() {
     size = 16;
     h = new char *[size];
     count = 0;
  }
  ~minMsgHeap() {
     delete [] h;
  }
  inline int length() const { return count; }
  inline int isEmpty() { return (count == 0); }
  void expand() {
    char **oldh = h;
    int oldcount = count;
    size *=2;
    h = new char *[size];
    count = 0;
    for (int i=0; i<oldcount; i++) enq(oldh[i]);
    delete [] oldh;
  }
  void enq(char *m) {
//CmiPrintf("enq %p\n", m);
      int current;

      if (count < size) {
        h[count] = m;
        current = count;
        count++;
      } else {
        expand();
        enq(m);
        return;
      }

      int parent = (current - 1)/2;
      while (current != 0)
        {
          if (CmiBgMsgRecvTime(h[current]) < CmiBgMsgRecvTime(h[parent]))
            {
              swap(current, parent);
              current = parent;
              parent = (current-1)/2;
            }
          else
            break;
        }
  }

  char *deq() {
//CmiPrintf("deq \n");
    if (count == 0) return 0;

    char *tmp = h[0];
    int best;

    h[0] = h[count-1];
    count--;

    int current = 0;
    int c1 = 1; int c2 = 2;
    while (c1 < count)
    {
      if (c2 >= count)
	best = c1;
      else
	{
	  if (CmiBgMsgRecvTime(h[c1]) < CmiBgMsgRecvTime(h[c2]))
	    best = c1;
	  else
	    best = c2;
	}
      if (CmiBgMsgRecvTime(h[best]) < CmiBgMsgRecvTime(h[current]))
	{
	  swap(best, current);
	  current = best;
	  c1 = 2*current + 1;
	  c2 = c1 + 1;
	}
      else
	break;
    }
    return tmp;
  }
  char * operator[](size_t n)
  {
//CmiPrintf("[] %d\n", n);
    return h[n];
  }
};


template<class T> class bgQueue;


#endif
