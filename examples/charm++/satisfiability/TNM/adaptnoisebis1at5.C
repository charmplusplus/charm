#define invPhi 10
#define invTheta 5

int lastAdaptFlip, lastBest, AdaptLength, NB_BETTER;

int initNoise() {
  lastAdaptFlip=0;
  lastBest = MY_CLAUSE_STACK_fill_pointer;
  NOISE=0; LNOISE=0; NB_BETTER=0;
  AdaptLength=NB_CLAUSE / invTheta;
}

void adaptNoveltyNoise(int flip) {
  if ((flip - lastAdaptFlip) > AdaptLength) {
    NOISE += (int) ((100 - NOISE) / invPhi);
    LNOISE= (int) NOISE/10;
    lastAdaptFlip = flip;      
    // NB_BETTER=0;
    lastBest = MY_CLAUSE_STACK_fill_pointer;
  } 
  else if (MY_CLAUSE_STACK_fill_pointer < lastBest) {
    //  NB_BETTER++;
    //  if (NB_BETTER>1) {
      NOISE -= (int) (NOISE / invPhi / 2);
      LNOISE= (int) NOISE/10;
      lastAdaptFlip = flip;
      lastBest = MY_CLAUSE_STACK_fill_pointer;
      //   NB_BETTER=0;
      // }
  }
}
