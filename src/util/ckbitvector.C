#include <string.h>
#include <charm++.h>
#include "ckbitvector.h"

/* *********************************************************************
 *
 *
 * ********************************************************************* */
bool CkBitVector::growToFit(unsigned int n) {
  if ( n > usedBits ) {
    int newUsedBits = n;
    int newDataLength = chunksForBits(newUsedBits);
  
    if ( newDataLength > dataLength ) {
      unsigned int *oldData = data;
      data = new unsigned int[newDataLength];
      bzero(data, newDataLength * sizeof(unsigned int));
    
      // Copy the old data back into it in it's old place
      for ( int i = 0 ; i < dataLength ; i++ ) {
        data[i] = oldData[i];
      }
    
      dataLength = newDataLength;
      delete [] oldData;
    }

    usedBits = n;

    return true;
  } else {
    return false;
  }
}

CkBitVector::CkBitVector()
  : usedBits(0), dataLength(1) {
  data = new unsigned int[dataLength];
  data[0] = 0;
}

CkBitVector::CkBitVector(const CkBitVector &b) {
  usedBits = b.usedBits;
  dataLength = b.dataLength;
  data = new unsigned int[dataLength];
  memcpy(data, b.data, dataLength*sizeof(unsigned int));
}


CkBitVector::CkBitVector(unsigned int size) : usedBits(size) {
  dataLength = chunksForBits(usedBits);
  data = new unsigned int[dataLength];
  bzero(data, dataLength*sizeof(unsigned int));
}

// Here since we are passing in ints and storing things as ints we can't
// ever exceed the size of the thing by an int.
CkBitVector::CkBitVector(unsigned int value, unsigned int choices)
  : usedBits(0), dataLength(1) {
  data = new unsigned int[dataLength];
  data[0] = 0;

  if ( value >= choices ) {
    ckout << "The value may not be bigger than the choices!" << endl;
    return;
  }

  if ( choices > 0x80000000 ) {
    ckout << "Can't create a bitvector with more choices than "
          << 0x80000000 << " due to the limits of integer arguments."
	  << endl;
  }

  usedBits = log2(choices);
  data[0] = value;
};


CkBitVector::~CkBitVector() {
  if ( data != NULL ) {
    delete [] data;
  }
}


CkBitVector & CkBitVector::operator=(const CkBitVector &b) {
  if ( this != &b ) {
    // If I had data allocated, free it
    if ( data != NULL ) {
      delete [] data;
    }
  
    // Copy over the other's data
    usedBits = b.usedBits;
    dataLength = b.dataLength;
    data = new unsigned int[dataLength];
    memcpy(data, b.data, dataLength*sizeof(unsigned int));
  }

  return *this;
}


CkBitVector & CkBitVector::Clear(unsigned int n) {
  unsigned int block = n/(sizeof(unsigned int) * 8),
               bit = n%(sizeof(unsigned int) * 8);

  growToFit(n);					// Expand if need be
  data[block] = data[block] & ~(0x1 << bit);	// Clear that bit
  return *this;					// Return the result
}


CkBitVector & CkBitVector::Set(unsigned int n) {
  unsigned int block = n/(sizeof(unsigned int) * 8),
               bit = n%(sizeof(unsigned int) * 8);

  growToFit(n);					// Expand if need be
  data[block] = data[block] | (0x1 << bit);	// Clear that bit
  return *this;					// Return the result
}


bool CkBitVector::Test(unsigned int n) const {
  unsigned int block = n/(sizeof(unsigned int) * 8),
               bit = n%(sizeof(unsigned int) * 8);

  if ( n > usedBits ) {
    return false;
  } else {
    return (bool)((data[block] >> bit) & 0x1);	// Return that bit
  }
}


CkBitVector & CkBitVector::Union(const CkBitVector &b) {
  // If b is bigger, grow to be at least as large as it is.
  growToFit(b);

  // Now or each chunk of the other operand into it. The only
  // kicker is to watch and make sure none of the bits above
  // the length of the other CkBitVector is corrupted and
  // contaminates this bitvector.
  int uB = b.usedBits;
  for ( int i = 0 ; i < b.dataLength ; i++ ) {
    int chunk = b.data[i] & maskBlock(uB);
    uB -= chunkSize();
    data[i] = data[i] | chunk;
  }

  return *this;
}


CkBitVector & CkBitVector::Intersection(const CkBitVector &b) {
  // If b is bigger, grow to be at least as large as it is.
  growToFit(b);

  // Now and each chunk of the other operand into it. The only
  // kicker is to watch and make sure none of the bits above
  // the length of the other CkBitVector is corrupted and
  // contaminates this bitvector.
  int uB = b.usedBits;
  for ( int i = 0 ; i < b.dataLength ; i++ ) {
    int chunk = b.data[i] & maskBlock(uB);
    uB -= chunkSize();
    data[i] = data[i] & chunk;
  }

  return *this;
}


CkBitVector & CkBitVector::Difference(const CkBitVector &b) {
  // If b is bigger, grow to be at least as large as it is.
  growToFit(b);

  // strip each chunk of the other operand off the result
  // IF the result exists there. If it doesn't, don't create
  // it to set it to 0.
  for ( int i = 0 ; i < b.dataLength ; i++ ) {
    int chunk = ~b.data[i];
    data[i] = data[i] & chunk;
  }

  return *this;
}


CkBitVector & CkBitVector::Concat(const CkBitVector &b) {
  CkBitVector tmp = b;		// Create a temporary bit vector in case the
  				// user did b.Concat(b) so we don't get
				// screwed up.
  ShiftUp(tmp.usedBits);	// Grow this vector by the size of the other
  return Union(tmp);		// Now just union the two together
}


CkBitVector & CkBitVector::ShiftUp(unsigned int n) {
  // If they asked me to grow by more than a chunk, do that first
  if ( n >= chunkSize() ) {
    unsigned int *oldData = data,
                 oldDataLength = dataLength,
                 integralN = n - (n % chunkSize());

    usedBits += integralN;
    dataLength += integralN/chunkSize();
    data = new unsigned int[dataLength];
    bzero(data, dataLength*sizeof(unsigned int));
    memcpy(data+(dataLength-oldDataLength), oldData, oldDataLength*sizeof(unsigned int));

    n -= integralN;
    delete [] oldData;
  }

  if ( n > 0 ) {
    unsigned int *oldData = data,
                 oldDataLength = dataLength;

    usedBits += n;
    dataLength = chunksForBits(usedBits);
    data = new unsigned int[dataLength];
    bzero(data, dataLength*sizeof(unsigned int));

    // Now copy each chunk over, doing the shifts
    for ( int i = 0 ; i < oldDataLength ; i++ ) {
      // Copy direct to direct
      data[i] |= oldData[i] << n;
      // Copy indirect from a chunk lower if the higher chunk exists
      if ( i+1 < dataLength ) {
	data[i+1] |= oldData[i] >> (chunkSize() - n);
      }
    }
  }

  return *this;
}


CkBitVector & CkBitVector::ShiftDown(unsigned int n) {
  // If they asked me to shrink by more than a unsigned int, do
  // a quick shift first to move down by however many unsigned
  // int chunks they asked to shrink by, then do the messy bit
  // shifting within chunks.
  if ( n >= chunkSize() ) {
    unsigned int *oldData = data,
                 oldDataLength = dataLength,
		 integralN = n - (n % chunkSize());

    usedBits -= integralN;
    dataLength -= integralN/chunkSize();
    data = new unsigned int[dataLength];
    bzero(data, dataLength*sizeof(unsigned int));
    memcpy(data, oldData+(oldDataLength-dataLength), dataLength*sizeof(unsigned int));
    delete [] oldData;

    n -= integralN;
  }

  if ( n > 0 ) {
    // Create an exact copy of the data
    unsigned int *oldData = data,
                 oldDataLength = dataLength;

    // Create the new smaller area we'll copy stuff into
    usedBits -= n;
    dataLength = chunksForBits(usedBits);
    data = new unsigned int[dataLength];
    bzero(data, dataLength*sizeof(unsigned int));

    // Now copy each chunk over, doing the shifts
    for ( int i = 0 ; i < dataLength ; i++ ) {
      // Copy direct to direct
      data[i] |= oldData[i] >> n;
      // Copy indirect from a chunk higher down if it exists
      if ( i+1 < oldDataLength ) {
	data[i] |= oldData[i+1] << (chunkSize() - n);
      }
    }
  }

  return *this;
}


CkBitVector & CkBitVector::Complement() {
  int uB = usedBits;
  for ( int i = 0 ; i < dataLength ; i++ ) {
    data[i] = ~data[i] & maskBlock(uB);
    uB -= chunkSize();
  }
  return *this;
}


bool CkBitVector::operator==(CkBitVector const &b) const {
  if ( this->usedBits != b.usedBits ||
       this->dataLength != b.dataLength ) {
    return false;
  }

  for ( int i = 0 ; i < this->dataLength ; i++ ) {
    if ( this->data[i] != b.data[i] ) {
      return false;
    }
  }

  return true;
}


bool CkBitVector::operator<(CkBitVector const &b) const {
  if ( *this == b ) {
    return false;
  }

  if ( this->usedBits > b.usedBits ||
       this->dataLength > b.dataLength ) {
    return false;
  }

  for ( int i = 0 ; i < this->dataLength ; i++ ) {
    if ( this->data[i] > b.data[i] ) {
      return false;
    }
  }

  return true;
}


bool CkBitVector::operator>(CkBitVector const &b) const {
  if ( *this == b ) {
    return false;
  }

  if ( this->usedBits < b.usedBits ||
       this->dataLength < b.dataLength ) {
    return false;
  }

  for ( int i = 0 ; i < this->dataLength ; i++ ) {
    if ( this->data[i] < b.data[i] ) {
      return false;
    }
  }

  return true;
}


char * CkBitVector::string() {
  char *buff = new char[usedBits+1];

  for ( unsigned int i = 0 ; i < usedBits ; i++ ) {
    buff[usedBits-1-i] = Test(i) ? '1' : '0';
  }

  buff[usedBits] = '\0';
  return buff;
}


/*
void CkBitVector::setEO( CkEntryOptions *eo ) {
  eo->setPriority(usedBits, data);
}
*/


unsigned int CkBitVector::log2 ( unsigned int val ) {
  unsigned int log = 0;
  if ( val != 0 ) {
    while ( val > (1<<log) ) { log++; }
  }
  return log;
}

unsigned int CkBitVector::maskBlock ( unsigned int length ) {
  unsigned int mask = 0;
  if ( length > chunkSize() ) {
    length = chunkSize();
  }
  while ( length > 0 ) {
    mask = mask << 1;
    mask |= 0x1;
    length--;
  }
  return mask;
}

CkOutStream& operator<< (CkOutStream& ckos, CkBitVector const b ) {
  char *buff = new char[b.usedBits+1];

  for ( unsigned int i = 0 ; i < b.usedBits ; i++ ) {
    buff[b.usedBits-1-i] = b.Test(i) ? '1' : '0';
  }

  buff[b.usedBits] = '\0';
  
  ckos << buff;
  delete [] buff;
}
