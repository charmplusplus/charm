#include "ampiimpl.h"

// Default ampi_setup
// this file is linked only when a user-defined ampi_setup function
// is not detected.

extern "C" void ampi_main(int, char**);
extern "C" void ampi_setup(void) { ampi_register_main(ampi_main); }

