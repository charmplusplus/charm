#include <stdio.h>
#include <stdlib.h>
#include "tetmesh.h"
#include "charm.h" //For CkAbort

int main(int argc,char *argv[]) {
	if (argc<2) {printf("Usage: test_noboite <.noboite file>\n"); exit(1);}
	TetMesh msh;
	readNoboite(fopen(argv[1],"r"),msh);
	print(msh);
	return 0;
}
