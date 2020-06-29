// Provide a stub entry point so the program will link without any special effort.

#include <stdio.h>

#ifdef main
# undef main
#endif
int main()
{
  fprintf(stderr, "Do not run this binary directly!\n");
  return 1;
}
