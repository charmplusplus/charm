
extern "C" {
#include "converse.h"
void InitializeCharm(int argc, char **argv);
void StartCharm(int argc, char **argv, void *whoknows);
}

void charm_init(int argc, char **argv)
{
  InitializeCharm(argc, argv);
  StartCharm(argc, argv, (void *)0);
}

main(int argc, char **argv)
{
  ConverseInit(argc, argv, charm_init,0,0);
}
