#include "converse.h"

user_main(argc, argv)
int argc;
char *argv[];
{
  CmiPrintf("user_main probably not currently working\n");
  /*  ConverseInit(argv); */

  pvmc_init_bufs();
#ifdef PVM_DEBUG
  CmiPrintf("Calling pvmc_init_comm()\n");
#endif

  pvmc_init_comm();

#ifdef PVM_DEBUG
  CmiPrintf("Starting pvmc_user_main\n");
#endif

  pvmc_user_main(argc,argv);
}

