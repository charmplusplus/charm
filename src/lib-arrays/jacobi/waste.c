#include <stdio.h>

main()
{
  double j=2.3456;
  while (1) {
	j *=j+2.0;
	j *=j+2.0;
	j =j*j;
  }
}
