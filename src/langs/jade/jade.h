// emacs mode line -*- mode: c++; tab-width: 4 -*-

/** Naming conventions:

  JADE_ is the prefix for any Jade calls exposed to the Jade user as
  part of the Jade API.

  Jade packages, such as MSA, which are part of the Jade language,
  should follow the above convention, but for now let it be.

  _JADEIDENT_ is the prefix for any compiler generated symbols not
  exposed to the user as part of the API.

*/

#ifndef JADE_H
#define JADE_H

// This part of the .h file is for Jade users to read too.

// The Jade user needs to know the MSA API.  But, strictly speaking,
// there should be a java file with the MSA API.
#include "msa/msa.h"

// This part is the .h file used by the compiler generated code.  The
// Jade user should never see it.  I will put it into a separate file
// when I have the time.

#include "JArray.h"

#define _JADE_MIN(a,b) (((a)<(b))?(a):(b))

/// Fast, fixed-size bitvector class, by Orion.
template <unsigned int NUM_BITS>
class _jade_fixedlength_bitvector {
public:
	/// Data type used to store actual bits in the vector.
	typedef unsigned long store_t;
	enum { store_bits=8*sizeof(store_t) };
	
	/// Number of store_t's in our vector.
	enum { len=(NUM_BITS+(store_bits-1))/store_bits };
	store_t store[len];
	
	/// Fill the entire vector with this value.
	void fill(store_t s) {
		for (int i=0;i<len;i++) store[i]=s;
	}
	void clear(void) {fill(0);}
	_jade_fixedlength_bitvector() {clear();}
	
	/// Set-to-1 bit i of the vector.
	void set(unsigned int i) { store[i/store_bits] |= (1lu<<(i%store_bits)); }
	/// Clear-to-0 bit i of the vector.
	void clear(unsigned int i) { store[i/store_bits] &= ~(1lu<<(i%store_bits)); }
	
	/// Return the i'th bit of the vector.
	bool get(unsigned int i) { return (store[i/store_bits] & (1lu<<(i%store_bits))); }

//     virtual void pup(PUP::er &p){p(store,len);};
};

#endif
