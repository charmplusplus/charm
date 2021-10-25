

#include "NDMeshStreamer.h"
#include "NDMeshStreamer.def.h"

//below code initializes the templated static variables from the header
CkArrayIndex1D TramBroadcastInstance<CkArrayIndex1D>::value=TRAM_BROADCAST;

CkArrayIndex2D TramBroadcastInstance<CkArrayIndex2D>::value=CkArrayIndex2D(TRAM_BROADCAST,TRAM_BROADCAST);

CkArrayIndex3D TramBroadcastInstance<CkArrayIndex3D>::value=CkArrayIndex3D(TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST);

CkArrayIndex4D TramBroadcastInstance<CkArrayIndex4D>::value=CkArrayIndex4D(TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST);

CkArrayIndex5D TramBroadcastInstance<CkArrayIndex5D>::value=CkArrayIndex5D(TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST);

CkArrayIndex6D TramBroadcastInstance<CkArrayIndex6D>::value=CkArrayIndex6D(TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST,TRAM_BROADCAST);

//Below code picks the appropriate TRAM_BROADCAST index value
CkArrayIndex& TramBroadcastInstance<CkArrayIndex>::value(int dims) {
  switch(dims) {
    case 1: return TramBroadcastInstance<CkArrayIndex1D>::value;
    case 2: return TramBroadcastInstance<CkArrayIndex2D>::value;
    case 3: return TramBroadcastInstance<CkArrayIndex3D>::value;
    case 4: return TramBroadcastInstance<CkArrayIndex4D>::value;
    case 5: return TramBroadcastInstance<CkArrayIndex5D>::value;
    case 6: return TramBroadcastInstance<CkArrayIndex6D>::value;
    default: CmiAbort("TRAM only supports 1-6D arrays\n");
  }
}
