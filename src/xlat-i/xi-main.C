
#include "xi-symbol.h"

extern void Generate(char *interfacefile) ;

main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cout << "Usage : " << argv[0] << " module.ci" << endl ;
    exit(1) ;
  }
  Module *m = Parse(argv[1]) ;
  Generate(argv[1]) ;
  return 0 ;
}

