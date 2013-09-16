#ifndef __UIUC_CS_CHARM_CKBITVECTOR_H
#define __UIUC_CS_CHARM_CKBITVECTOR_H

#include "ckstream.h"

/* ************************************************************************
 *
 * 0 indicates the least important bit of the bitvector
 * n indicates the most inportant bit of the bitvector
 *
 * the n'th bit sits at the highest bit of the first integer.
 * the 0'th bit sits at the lowest bit of the last integer.
 *
 * ************************************************************************ */

typedef CmiUInt4 prio_t;

class CkBitVector {
 protected:
  prio_t usedBits;
  prio_t *data;

 protected:
  static prio_t chunkBits() { return chunkSize()*8; }
  static prio_t chunkSize() { return sizeof(prio_t); }
  static prio_t chunks(prio_t n) { return (n + chunkBits()-1) / chunkBits(); }
  prio_t offset(prio_t bit) const { return chunks(usedBits-bit)-1; }
  prio_t mask(prio_t bit) const {
    unsigned int shift = (chunkBits()-(usedBits%chunkBits())+(bit%chunkBits()));
    shift %= chunkBits();
    return (((prio_t)0x1)<<shift); 
  }

 public:
  static prio_t ilog2(prio_t val) {
    prio_t log = 0u;
    if ( val != 0u ) {
      while ( val > (1u<<log) ) { log++; }
    }
    return log;
  }

 protected:
  void wipeData();

 public:
  int Length() const { return (int)usedBits; }

 public:
  CkBitVector();
  CkBitVector(const CkBitVector &b);
  CkBitVector(prio_t bits);
  CkBitVector(prio_t value, prio_t choices);

  ~CkBitVector();

  CkBitVector & operator=(const CkBitVector &b);

  // Bit operations. 0 is the least significant bit, and 32 (+) is the
  // most significant bit.
  CkBitVector & Zero();
  CkBitVector & Invert();
  CkBitVector & Clear(prio_t bit);
  CkBitVector & Set(prio_t bit);
  bool Test(prio_t bit) const;

  // Shift down and shift up shift the bits in the bit vector by bit
  // bits around. The bits in the vector are moved up or down the
  // specified amount. The size of the bit vector does not change
  CkBitVector & ShiftDown(prio_t bits);
  CkBitVector & ShiftUp(prio_t bits);

  // Change the size of the bit vector
  CkBitVector & Resize(prio_t bits);

  // Union, Intersection, Difference
  CkBitVector & Union(CkBitVector const &b);
  CkBitVector & Intersection(CkBitVector const &b);
  CkBitVector & Difference(CkBitVector const &b);

  // Concatenate two bit vectors together
  CkBitVector & Concat(CkBitVector const &b);

  // Comparison operators
  int  Compare(const CkBitVector &b) const;
  bool operator==(const CkBitVector &b) const {   if(Compare(b) == 0) return true; else return false; } // HERE
  bool operator!=(const CkBitVector &b) const { return !(*this==b); }
  bool operator<(const CkBitVector &b) const { if(Compare(b) == -1) return true; else return false; } // HERE
  bool operator<=(const CkBitVector &b) const { return (*this==b||*this>b); }
  bool operator>(const CkBitVector &b) const { if(Compare(b) == 1) return true; else return false;} // HERE
  bool operator>=(const CkBitVector &b) const { return (*this==b||*this<b); }

  // Print the bit vector to either output stream type
  friend CkOutStream & operator<< (CkOutStream &ckos, CkBitVector const b);
  friend CkErrStream & operator<< (CkErrStream &ckes, CkBitVector const b);

  // And for charm
  void pup(PUP::er &p);

  // For debugging in megatest
#ifdef DEBUGGING
  CmiUInt4 * getData() { return data; }
  unsigned int getDataLength() { return chunks(usedBits); }
#endif

  friend class CkEntryOptions;
};

PUPmarshall(CkBitVector)

#endif /* __UIUC_CS_CHARM_CKBITVECTOR_H */
