#include <iostream.h>
#include "idlString.hh"

main()
{
  IdlString a("Hello");
  IdlString b;
  b = " ";
  IdlString c("World\n");

  a+b+c;
  cout << a << endl;
  a +=b+c;
  cout << a;
  cout << b;
}
