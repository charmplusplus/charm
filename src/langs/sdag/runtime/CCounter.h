/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CCounter_H_
#define _CCounter_H_

class CCounter {
  private:
    unsigned int count;
  public:
    CCounter(int c) : count(c) {}
    CCounter(int first, int last, int stride) {
      count = ((last-first)/stride)+1;
    }
    void decrement(void) {count--;}
    int isDone(void) {return (count==0);}
};

#endif
