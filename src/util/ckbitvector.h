#ifndef __UIUC_CHARM_BITVECTOR_H
#define __UIUC_CHARM_BITVECTOR_H

//#include <charm++.h>
#include <ckstream.h>

/* *********************************************************************
 *
 *
 * ********************************************************************* */
class CkBitVector {
 protected:
  unsigned int usedBits;
  unsigned int *data;
  int dataLength;

  bool growToFit(unsigned int n);
  bool growToFit(const CkBitVector &b) { return growToFit(b.usedBits); }
  static unsigned int chunkSize() { return sizeof(unsigned int)*8; }
  static unsigned int chunksForBits(unsigned int n) {
    return ( n + (chunkSize() - 1) ) / chunkSize();
  }

 public:
  CkBitVector();
  CkBitVector(const CkBitVector &b);
  CkBitVector(unsigned int size);
  CkBitVector(unsigned int value, unsigned int choices);

  ~CkBitVector();

  CkBitVector & operator=(const CkBitVector &b);

  // Get the value at a bit, set the value of a bit to true, or
  // clear the value of a bit (set to false)
  CkBitVector & Clear(unsigned int n);
  CkBitVector & Set(unsigned int n);
  bool Test(unsigned int n) const;
  CkBitVector & Complement();

  // Perhaps these belong as union and intersection instead,
  // as they are "bitwise and" and "bitwise or".
  // Where does that leave + and - and * tho?
  CkBitVector operator|(unsigned int n) { CkBitVector r = *this; return r.Set(n); }
  CkBitVector & operator|=(unsigned int n) { return Set(n); }
  CkBitVector operator&(unsigned int n) { CkBitVector r = *this; return r.Clear(n); }
  CkBitVector & operator&=(unsigned int n) { return Clear(n); }
  CkBitVector operator~() { CkBitVector r = *this; return r.Complement(); }

  // union, intersect, difference, concat
  CkBitVector & Union(const CkBitVector &b);
  CkBitVector & Intersection(const CkBitVector &b);
  CkBitVector & Difference(const CkBitVector &b);
  CkBitVector & Concat(const CkBitVector &b);

  // These are aliases for union, intersect, and diff
  CkBitVector operator+(const CkBitVector &b) { CkBitVector r = *this; return r.Union(b); }
  CkBitVector & operator+=(const CkBitVector &b) { return Union(b); }
  CkBitVector operator-(const CkBitVector &b) { CkBitVector r = *this; return r.Difference(b); }
  CkBitVector & operator-=(const CkBitVector &b) { return Difference(b); }
  CkBitVector operator*(const CkBitVector &b) { CkBitVector r = *this; return r.Intersection(b); }
  CkBitVector & operator*=(const CkBitVector &b) { return Intersection(b); }

  // Expand and shrink the bitvector
  CkBitVector & ShiftUp(unsigned int n);
  CkBitVector & ShiftDown(unsigned int n);

  // These make sense
  CkBitVector operator<<(unsigned int n) { CkBitVector r = *this; return r.ShiftUp(n); }
  CkBitVector & operator<<=(unsigned int n) { return ShiftUp(n); }
  CkBitVector operator>>(unsigned int n) { CkBitVector r = *this; return r.ShiftDown(n); }
  CkBitVector & operator>>=(unsigned int n) { return ShiftDown(n); }

  // These are fine
  bool operator==(const CkBitVector &b) const;
  bool operator!=(const CkBitVector &b) const { return !(*this == b); }
  bool operator<(const CkBitVector &b) const;
  bool operator<=(const CkBitVector &b) const { return *this==b || *this<b; }
  bool operator>(const CkBitVector &b) const;
  bool operator>=(const CkBitVector &b) const { return *this==b || *this>b; }


  // I'd actually like to just be able to print it out directly to a
  // iostream rather than bitvector->string() it. Should fix this.
  char * string();

  // This should probably be in CkEntryOptions not in here but until
  // the class is put in the core it'll have to hold here.
//  void setEO(CkEntryOptions *eo);

  // This is an integer log base 2 function. Some glibcs have them,
  // some don't. Not a very bright one, it just tests for less than
  // 2^i for i = 0 .. sizeof(int)*4. It takes the ceiling of the
  // true floating point result log2(x) would return.
  //
  // Aka, it takes a number and tells you how many bits you need to
  // represent that number.
  static unsigned int log2(unsigned int input);

  static unsigned int maskBlock(unsigned int s);

#ifdef DEBUGGING
  unsigned int *getData() { return data; }
  unsigned int getDataLength() { return dataLength; }
  unsigned int getUsedBits() { return usedBits; }
#endif

  friend CkOutStream & operator<< (CkOutStream& ckos, CkBitVector const b );
  friend class CkEntryOptions;
};


#endif /* __UIUC_CHARM_BITVECTOR_H */
