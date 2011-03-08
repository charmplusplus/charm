#include "charm++.h"

/*
 A (partial) array that "boomerangs" back to its original source.
 On the original PE, the array data is used in-place.
 On non-original PEs, the array data is dynamically allocated.
*/
template <typename T>
class Boomarray : public CkConditional {
 public:
  /* Constructor used on *original* data.
     This class will keep a persistent pointer to this data,
     and use this data as storage whenever possible.
     Beware!  This will crash if src migrates, checkpoints, etc!
  */
  Boomarray(int count_, T *src_)
    :count(count_), ptr(src_),
    srcnode(CkMyNode()), srcptr(src_) {
    DebugPrintf("Boomarray constr\n");
  }

    /* Constructor used on *target* PE (migration) */
    Boomarray() {
      DebugPrintf("Boomarray migration\n");
    }

    /* Destructor (frees memory on client) */
    ~Boomarray() { DebugPrintf("~Boomarray\n"); if (ptr!=srcptr) free(ptr);}

    /* Access the array */
    double& operator[](int index) {return ptr[index];}

    void pup(PUP::er &p) {
      DebugPrintf1("Boomarray::pup %s\n",p.isPacking()?"packing":(p.isSizing()?"sizing":"unpacking"));
      /* Move our housekeeping metadata */
      p|count;
      p|srcnode;
      p|*(long *)&srcptr; /* pack as a long */
      if (p.isUnpacking())
        { /* gotta set up ptr */
          if (CmiMyNode()==srcnode) /* back to same old processor */
            ptr=srcptr;
          else /* living on a new processor */
            ptr=(T *)malloc(count*sizeof(T));
        }
      /* Move the underlying network data */
      p(ptr,count);
    }

    int size() { return count; }

 private:
    /* Length of data (doubles) */
    int count;
    /* Data to store (if not on original PE) */
    T *ptr;

    /* processor where the data originated */
    int srcnode;
    /* pointer on original PE where data returns to */
    T *srcptr;
};

