#include "ampiimpl.h"

// Default ampi_setup

#if AMPI_FORTRAN
#include "ampimain.decl.h"
#if CMK_FORTRAN_USES_ALLCAPS
extern "C" void AMPI_MAIN(int, char **);
extern "C" void AMPI_SETUP(void){AMPI_REGISTER_MAIN(AMPI_MAIN);}
#else
extern "C" void ampi_main_(int, char **);
extern "C" void ampi_setup_(void){ampi_register_main_(ampi_main_);}
#endif
#else
extern "C" void AMPI_Main(int, char **);
extern "C" void AMPI_Setup(void){AMPI_Register_main(AMPI_Main);}
#endif

