#ifndef _CKLISTS_H_
#define _CKLISTS_H_

#include <unistd.h>

class CkVector {
public:
  CkVector() { items = new void*[blk_sz]; in_use = 0; cur_sz = blk_sz; };
  CkVector(size_t n) { items = new void*[n]; in_use = 0; cur_sz = n;  };
  ~CkVector() { if (items) delete [] items; }

  void*& operator[](size_t n) { return items[n]; };

  void push_back(void* item) {
    if (cur_sz == in_use) grow_list();
    items[in_use] = item;
    in_use++;
  };

  size_t size() { return in_use; };

private:
  void grow_list() {
    void** old_items = items;
    cur_sz *= 2;
    items = new void*[cur_sz];
    for(int i=0; i < in_use; i++)
      items[i] = old_items[i];
    delete [] old_items;
  };
    
  enum { blk_sz = 1000 };		 
  void** items;
  int in_use;
  int cur_sz;
};

#endif
