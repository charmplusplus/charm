#include <stdio.h>
#include <stdlib.h>
#include "prismMesh.h"

// declaring these inline confuses the Intel C++ 7.1 compiler...
CkVector3d *PrismMesh::getPointArray(void) {
  return &(pts[0]);
}
const CkVector3d *PrismMesh::getPointArray(void) const {
  return &(pts[0]);
}
