#include <string.h> // for memset
#include "charm++.h"
#include "ckbitvector.h"

/* ************************************************************************
 * CkBitVector
 *
 * The CkBitVector class implements a bit vector in the same order that
 * charm expects it for the message priorities. The way that works is:
 *
 * Highest bit of the bit vector goes in the first integer's highest bit.
 * The next goes in the next, so on and so fourth until it hits the first
 * integer's lowest bit. The next bit in order is the highest bit of
 * the next integer. So you get a pattern like:
 *
 *  xx xx xx xx  xx xx xx xx  xx xx xx xx 
 *     Int 0        Int 1        Int 2
 *  0        32  0        32  0        32
 *            ^
 *   <<<<<<<<<
 *  v______________________
 *                         ^
 *                <<<<<<<<<
 *               v______________________
 *                                      ^
 *                             <<<<<<<<<
 *                            v
 *
 * Where you follow the path down reading bits where there is a ^, v, or <.
 *
 * ************************************************************************ */
// Construct an empty bit vector
CkBitVector::CkBitVector() : usedBits(0) {
}

// Construct a copy of a bit vector
CkBitVector::CkBitVector(const CkBitVector &b) : usedBits(b.usedBits) {
  data = b.data;
}

// Construct a bit vector of the specified size.
CkBitVector::CkBitVector(prio_t bits) : usedBits(bits) {
  if ( bits != 0 ) {
    data.resize(chunks(usedBits));
  }
}

// Construct a bit vector that's set to the given value out of the number
// of choices specified
CkBitVector::CkBitVector(prio_t value, prio_t choices) {
  // If the user gave me a higher value than the choices, whine at them.
  if ( value >= choices ) {
    CkAbort("User asked for a bit vector too large for the number of choices specified!");
  }

  // Okay they didn't. Figure out how many bits I need to represent the
  // choices
  usedBits = ilog2(choices);
  if ( usedBits != 0 ) {
    data.resize(chunks(usedBits));
    data[0] = value << (chunkBits() - usedBits);
  }
}

// Nuke the memory occupied by a CkBitVector
CkBitVector::~CkBitVector() {
}

// Clean out any memory I'm using.
void CkBitVector::wipeData() {
  data.clear();
  usedBits = 0;
}


// Copy a CkBitVector into this one.
CkBitVector & CkBitVector::operator=(const CkBitVector &b) {
  usedBits = b.usedBits;
  data = b.data;

  // Return the resultant bitvector
  return *this;
}

// Move a CkBitVector into this one.
CkBitVector & CkBitVector::operator=(CkBitVector &&b) {
  usedBits = b.usedBits;
  data = std::move(b.data);

  // Return the resultant bitvector
  return *this;
}


// Zero all the bits in a vector
CkBitVector & CkBitVector::Zero() {
  std::fill(data.begin(), data.end(), 0);

  return *this;
}

// Invert the bit vector
CkBitVector & CkBitVector::Invert() {
  // Invert all my bits. Only tricky one is the last block I need to mask
  // those bits beyond the end of the vector to make sure they don't get
  // set and corrupt an operation like shifting later.
  for ( prio_t & d : data ) {
    d = ~d;
  }
  if ( usedBits % chunkBits() != 0 ) {
    data.back() &= ((~(unsigned int)0)<<(chunkBits()-(usedBits%chunkBits())));
  }

  return *this;
}

// Clear the given bit in the bitvector
CkBitVector & CkBitVector::Clear(prio_t bit) {
  // If it is out of range, grow yourself and that much. You don't need to
  // Clear the bit because it comes cleared already
  if ( bit+1 > usedBits ) {
    Resize(bit+1);
    return *this;
  }

  // The bit exists, compute where we'll find it.
  prio_t index = offset(bit);
  prio_t bitMask = ~(mask(bit));

  // Twiddle that bit to clear
  data[index] &= bitMask;

  // Return the modified bitvector
  return *this;
}

// Set a bit
CkBitVector & CkBitVector::Set(prio_t bit) {
  // If it is out of range, grow yourself and that much then set as normal.
  if ( bit+1 > usedBits ) {
    Resize(bit+1);
  }

  // The bit exists, compute where we'll find it.
  prio_t index = offset(bit);
  prio_t bitMask = mask(bit);

  // Twiddle that bit to set
  data[index] |= bitMask;

  // Return the modified bitvector
  return *this;
}

// Is the bit given set?
bool CkBitVector::Test(prio_t bit) const {
  // If it is out of range it's obviously false
  if ( bit+1 > usedBits ) { return false; }

  // If it's in range, calculate it's chunk and offset
  prio_t index = offset(bit);
  prio_t bitMask = mask(bit);

//ckerr << bit << ": at offset " << index << " mask " << bitMask << endl;

  // Access that bit, check it versus the mask and return the result
  return ((data[index]&bitMask) != 0);
}



// ShiftDown and ShiftUp
CkBitVector & CkBitVector::ShiftUp(prio_t bits) {
  // If data is null then we have nothing to work on
  // Also abort if we got a dud shift (0)
  if ( bits == 0 ) { return *this; }

  // Shift by the whole and by the remainder at the same time
  prio_t whole = bits / chunkBits();
  prio_t rem = bits % chunkBits();
  for ( int i = 0 ; i < chunks(usedBits) ; i++ ) {
     if ( i+whole < chunks(usedBits) ) {
       data[i] = data[i+whole] << rem;

       if ( i+whole+1 < chunks(usedBits) ) {
         data[i] |= data[i+whole+1] >> (chunkBits()-rem);
       }
     } else {
       data[i] = 0;
     }
  }

  return *this;
}

CkBitVector & CkBitVector::ShiftDown(prio_t bits) {
  // If data is null then we have nothing to work on
  // Also abort if we got a dud shift (0)
  if ( bits == 0 ) { return *this; }

  // Shift by the whole and by the remainder at the same time
  int whole = bits / chunkBits();
  int rem = bits % chunkBits();
  for ( int i = chunks(usedBits)-1 ; i >= 0 ; i-- ) {
     if ( i-whole >= 0 ) {
       data[i] = data[i-whole] >> rem;

       if ( i-whole-1 < chunks(usedBits) ) {
         data[i] |= data[i-whole-1] << (chunkBits()-rem);
       }
     } else {
       data[i] = 0;
     }
  }

  return *this;
}



// Resize
// For growth, resize will copy the bits back where they were (into the lowest
// bits of the new larger bit vector).
// For shrinking, resize will copy the highest bits back into the vector.
CkBitVector & CkBitVector::Resize(prio_t bits) {
  // If we got asked to size ourself to our current size, just chunk out
  if ( bits == usedBits ) { return *this; }
  // Asked to empty ourselves?
  else if ( bits == 0 ) {
    wipeData();
    return *this;
  }
  // Asked to grow?
  else if ( bits > usedBits ) {
    prio_t shift = bits - usedBits;
    data.resize(chunks(bits));
    usedBits = bits;
    return ShiftDown(shift);
  }
  // Shrink?
  else if ( bits < usedBits ) {
    ShiftUp(usedBits - bits);
    data.resize(chunks(bits));
    usedBits = bits;
    return *this;
  }

  // This shouldn't ever be reached
  CkAbort("What in heck did you do!!?!?! CkBitVector error in Resize()!");
  return *this;
}



// Union this bit vector with another.
CkBitVector & CkBitVector::Union(CkBitVector const &b) {
  // CkBitVectors must be of the same size.
  if ( usedBits != b.usedBits ) {
    CkAbort("CkBitVector Union operands must be of the same length!");
  }

  // Union them into me
  for ( int i = 0 ; i < chunks(usedBits) ; i++ ) {
    data[i] |= b.data[i];
  }

  return *this;
}

// Intersect this bit vector with another.
CkBitVector & CkBitVector::Intersection(CkBitVector const &b) {
  // CkBitVectors must be of the same size.
  if ( usedBits != b.usedBits ) {
    CkAbort("CkBitVector Intersection operands must be of the same length!");
  }

  // Intersect them into me
  for ( int i = 0 ; i < chunks(usedBits) ; i++ ) {
    data[i] &= b.data[i];
  }

  return *this;
}

// Take the difference of this bit vector with another.
CkBitVector & CkBitVector::Difference(CkBitVector const &b) {
  // CkBitVectors must be of the same size.
  if ( usedBits != b.usedBits ) {
    CkAbort("CkBitVector Difference operands must be of the same length!");
  }

  // Take any bits they have set out of me
  for ( int i = 0 ; i < chunks(usedBits) ; i++ ) {
    data[i] &= ~(b.data[i]);
  }

  return *this;
}



// Concat two bitvectors together. To do this, we'll need to clone the
// one that is being copied into this, and shift it left/right a bit.
CkBitVector & CkBitVector::Concat(CkBitVector const &b) {
  // If I'm null, just copy b into me.
  if ( data.empty() ) {
    *this = b;
    return *this;
  }

  // I exist. Create a clone of b I can shift around.
  CkBitVector tmp(b);

  // Grow b to match me, then move it's bits down to where they need to be
  tmp.Resize(usedBits + b.usedBits);

  // Grow to hold me and b.
  unsigned int shiftBy = b.usedBits;
  Resize(usedBits + b.usedBits);
  ShiftUp(shiftBy);

  // Now I can union us and get the result of the concatenation.
  Union(tmp);

  return *this;
}



// Spit out the bit vector in chunk-sized chunks
CkOutStream& operator<< (CkOutStream& ckos, CkBitVector const b ) {
  if ( !b.data.empty() ) {
    char *buff = new char[b.usedBits+1];
    for ( int i = b.usedBits-1 ; i >= 0 ; i-- ) {
      buff[(b.usedBits-1)-i] = (b.Test(i) ? '1' : '0');
    }
    buff[b.usedBits] = '\0';
    ckos << buff;
    delete[] buff;
  }

  return ckos;
}

CkErrStream& operator<< (CkErrStream& ckes, CkBitVector const b ) {
  if ( !b.data.empty() ) {
    char *buff = new char[b.usedBits+1];
    for ( int i = b.usedBits-1 ; i >= 0 ; i-- ) {
      buff[(b.usedBits-1)-i] = (b.Test(i) ? '1' : '0');
    }
    buff[b.usedBits] = '\0';
    ckes << buff;
    delete[] buff;
  }

  return ckes;
}

int CkBitVector::Compare(const CkBitVector &b) const
{
    int result = 0;
    int length, i;
    if(usedBits > b.usedBits)
    {
        result = 1;
        length = chunks(b.usedBits);
    }
    else if (usedBits < b.usedBits)
    {
        result = -1;
        length = chunks(usedBits);
    }
    else
    {
        result = 0;
        length = chunks(usedBits);
    }

    for(i=0; i<length; i++)
    {
        if(data[i] > b.data[i])
        {
            result = 1;
            break;
        }else if (data[i] < b.data[i])
        {
            result = -1;
            break;
        }
    }
    return result;
}

// Pack and unpack this bugger!
void CkBitVector::pup(PUP::er &p) {
  p|usedBits;
  p|data;
}
