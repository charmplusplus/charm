#include "ampiimpl.h"

// Default ampi_setup
// this file is linked only when a user-defined ampi_setup function
// is not detected.

#include <string.h> // for strlen
// default module name
#define D "default"

extern "C" void ampi_main(int, char**);
extern "C" void 
ampi_setup(void) 
{ 
  ampi_register_main(ampi_main, D, strlen(D)); 
}

