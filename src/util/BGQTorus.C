#include "BGQTorus.h"

#if CMK_BLUEGENEQ

#include "spi/include/kernel/process.h"
#include "spi/include/kernel/location.h"
#include <firmware/include/personality.h>

int *bgq_localNodes = NULL;
CmiNodeLock bgq_lock;
int bgq_isLocalSet = 0;

void bgq_topo_init() {
  static int init_done = 0;
  if (!init_done) {
    bgq_lock = CmiCreateLock();
    init_done = 1;
  }
}

void bgq_topo_free() { 
  CmiLock(bgq_lock);
  if(bgq_localNodes) free(bgq_localNodes);
  bgq_isLocalSet = 0;
  CmiUnlock(bgq_lock);
}

void bgq_topo_reset() {
  CmiLock(bgq_lock);
  bgq_isLocalSet = 0;
  CmiUnlock(bgq_lock);
}

BGQTorusManager::BGQTorusManager() {
  order[0] = 5;
  order[1] = 4;
  order[2] = 3;
  order[3] = 2;
  order[4] = 1;
  order[5] = 0;

  int numPes = CmiNumPesGlobal();
  procsPerNode = Kernel_ProcessCount();
  thdsPerProc = CmiMyNodeSize();
  hw_NT = procsPerNode*thdsPerProc;

  Personality_t pers;
  Kernel_GetPersonality(&pers, sizeof(pers));

  hw_NA = pers.Network_Config.Anodes;
  hw_NB = pers.Network_Config.Bnodes;
  hw_NC = pers.Network_Config.Cnodes;
  hw_ND = pers.Network_Config.Dnodes;
  hw_NE = pers.Network_Config.Enodes;

  unsigned int isFile = 0;
  Kernel_GetMapping(10, mapping, &isFile);
  if(!isFile) {
    for(int i = 0; i < 6 ; i++) {
      if(mapping[i] != 'T') {
        order[5 - i] = mapping[i] - 'A';
      } else {
        order[5 - i] = 5;
      }
    }
  }
  //printf("Mapping %d %d %d %d %d %d\n",order[0],order[1],order[2],order[3],order[4],order[5]);

  rn_NA = hw_NA;
  rn_NB = hw_NB;
  rn_NC = hw_NC;
  rn_ND = hw_ND;
  rn_NE = hw_NE;

  int max_t = 0;
  if(rn_NA * rn_NB * rn_NC * rn_ND * rn_NE != numPes/hw_NT) {
    rn_NA = rn_NB = rn_NC = rn_ND =rn_NE =0;
    int rn_NT=0;
    int min_a, min_b, min_c, min_d, min_e, min_t;
    min_a = min_b = min_c = min_d = min_e = min_t = (~(-1));
    int tmp_t, tmp_a, tmp_b, tmp_c, tmp_d, tmp_e;
    uint64_t numentries;
    BG_CoordinateMapping_t *coord;

    int nranks=numPes/thdsPerProc;
    coord = (BG_CoordinateMapping_t *) malloc(sizeof(BG_CoordinateMapping_t)*nranks);
    Kernel_RanksToCoords(sizeof(BG_CoordinateMapping_t)*nranks, coord, &numentries);

    for(int c = 0; c < nranks; c++) {
      tmp_a = coord[c].a;
      tmp_b = coord[c].b;
      tmp_c = coord[c].c;
      tmp_d = coord[c].d;
      tmp_e = coord[c].e;
      tmp_t = coord[c].t;

      if(tmp_a > rn_NA) rn_NA = tmp_a;
      if(tmp_a < min_a) min_a = tmp_a;
      if(tmp_b > rn_NB) rn_NB = tmp_b;
      if(tmp_b < min_b) min_b = tmp_b;
      if(tmp_c > rn_NC) rn_NC = tmp_c;
      if(tmp_c < min_c) min_c = tmp_c;
      if(tmp_d > rn_ND) rn_ND = tmp_d;
      if(tmp_d < min_d) min_d = tmp_d;
      if(tmp_e > rn_NE) rn_NE = tmp_e;
      if(tmp_e < min_e) min_e = tmp_e;
      if(tmp_t > rn_NT) rn_NT = tmp_t;
      if(tmp_t < min_t) min_t = tmp_t;
    }
    rn_NA = rn_NA - min_a + 1;
    rn_NB = rn_NB - min_b + 1;
    rn_NC = rn_NC - min_c + 1;
    rn_ND = rn_ND - min_d + 1;
    rn_NE = rn_NE - min_e + 1;
    procsPerNode = rn_NT - min_t + 1;
    hw_NT = procsPerNode * thdsPerProc;
    free(coord);
  }

  dimA = rn_NA;
  dimB = rn_NB;
  dimC = rn_NC;
  dimD = rn_ND;
  dimE = rn_NE;
  dimA = dimA * hw_NT;	// assuming TABCDE

  dims[0] = rn_NA;
  dims[1] = rn_NB;
  dims[2] = rn_NC;
  dims[3] = rn_ND;
  dims[4] = rn_NE;
  dims[5] = hw_NT;

  torus[0] = ((rn_NA % 4) == 0)? true:false;
  torus[1] = ((rn_NB % 4) == 0)? true:false;
  torus[2] = ((rn_NC % 4) == 0)? true:false;
  torus[3] = ((rn_ND % 4) == 0)? true:false;
  torus[4] = true;

  populateLocalNodes();
}

void BGQTorusManager::populateLocalNodes() {
  if(CmiNumPartitions() == 1) return;

  CmiLock(bgq_lock);
  if(bgq_isLocalSet) {
    CmiUnlock(bgq_lock);
    return;
  }

  if(bgq_localNodes == NULL)
    bgq_localNodes = (int *)malloc(CmiNumNodesGlobal()*sizeof(int));

  CmiAssert(bgq_localNodes != NULL);

  for(int i = 0; i < CmiNumNodesGlobal(); i++)
    bgq_localNodes[i] = -1;

  for(int i = 0; i < CmiNumNodes(); i++) {
    int a, b, c, d, e, t;
    int global;

    rankToCoordinates(CmiNodeFirst(i), a, b, c, d, e, t);
    global = CmiNodeOf(coordinatesToRank(a, b, c, d, e, t));

    bgq_localNodes[global] = i;
  }

  bgq_isLocalSet = 1;

  CmiUnlock(bgq_lock);
}

#endif

