#include "import.h"
void ParFUM_desharing(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_desharing"))->getMesh("ParFUM_desharing");
	mesh->clearSharedNodes();
}


void ParFUM_deghosting(int meshid){
	FEM_Mesh 	*mesh = (FEM_chunk::get("ParFUM_desharing"))->getMesh("ParFUM_desharing");
	mesh->clearGhostNodes();
	mesh->clearGhostElems();
}
