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

#endif
