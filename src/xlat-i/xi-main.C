
#include "xi-symbol.h"

extern void Generate(char *interfacefile) ;

const int SUCCESS = 0;
const int FAILURE = 1;

main(int argc, char *argv[])
{
	if ( argc != 2 ) {
		cout << "Usage : " << argv[0] << " module.ci" << endl ;
		exit(FAILURE) ;
	}

	Module *m = Parse(argv[1]) ;

	Generate(argv[1]) ;

	return SUCCESS ;
}

