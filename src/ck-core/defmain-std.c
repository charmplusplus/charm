
#include "converse.h"

void charm_init(argc, argv)
int argc;
char **argv;
{
  InitializeCharm(argc, argv);
  StartCharm(argc, argv, (void *)0);
}

main(argc, argv)
int argc;
char *argv[];
{
  ConverseInit(argc, argv, charm_init,0,0);
}
