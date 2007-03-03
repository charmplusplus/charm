
#ifndef  __PENCIL_API_H__
#define  __PENCIL_API_H__

#include "pencilfft.h"

inline void configureLineFFTInfo (LineFFTInfo *info,
				  int sizeX,  int sizeY,  int sizeZ,
				  int grainX, int grainY, int grainZ,
				  CkCallback  *kcallback,
				  LineFFTCompletion complid,
				  bool              normalize) 
{  
  info->sizeX   =  sizeX;
  info->sizeY   =  sizeY;
  info->sizeZ   =  sizeZ;
  
  info->grainX  =  grainX;
  info->grainY  =  grainY;
  info->grainZ  =  grainZ;
  
  if (kcallback) 
    info->kSpaceCallback = *kcallback;
  
  info->completionId = complid;
  info->normalize    = normalize;
}


inline void  createLineFFTArray (LineFFTInfo *info) {

  CkAssert (info->sizeX > 0);
  CkAssert (info->sizeY > 0);
  CkAssert (info->sizeZ > 0);
  
  CkAssert (info->grainX > 0);
  CkAssert (info->grainY > 0);
  CkAssert (info->grainZ > 0);
  
  info->xProxy = CProxy_LineFFTArray::ckNew();
  info->yProxy = CProxy_LineFFTArray::ckNew();
  info->zProxy = CProxy_LineFFTArray::ckNew();

  int x, y, z;

  double pe = 0.0;
  double stride = 
    (1.0 *CkNumPes() * info->grainZ * info->grainY)/
    (info->sizeZ * info->sizeY);  

  for (pe = 0.0, z = 0; z < (info->sizeZ)/(info->grainZ); z ++) {
    for (y = 0; y < (info->sizeY)/(info->grainY); y++) {
      if(pe >= CkNumPes()) pe = pe - CkNumPes();
      info->xProxy(y, z).insert(*info, (int) PHASE_X, (int) pe);
      pe +=  stride;
    }
  }
  info->xProxy.doneInserting();

  stride = 
    (1.0 *CkNumPes() * info->grainX * info->grainZ)/
    (info->sizeX * info->sizeZ);  

  for (pe=1.0, x = 0; x < (info->sizeX)/(info->grainX); x ++) {
    for (z = 0; z < (info->sizeZ)/(info->grainZ); z ++) {
      if(pe >= CkNumPes()) pe = pe - CkNumPes();
      info->yProxy(z, x).insert(*info, (int) PHASE_Y, (int) pe);
      pe += stride;
    }
  }
  info->yProxy.doneInserting();
  
  stride = 
    (1.0 *CkNumPes() * info->grainY * info->grainX)/
    (info->sizeY * info->sizeX);  

  for (pe=0.0, y = 0; y < (info->sizeY)/(info->grainY); y ++) {
    for (x = 0; x < (info->sizeX)/(info->grainX); x ++) {
      if(pe >= CkNumPes()) pe = pe - CkNumPes();
      info->zProxy(x, y).insert(*info, (int) PHASE_Z, (int) pe);
      pe += stride;
    }
  }
  info->zProxy.doneInserting();
}


inline void startLineFFTArray (LineFFTInfo *info) {
  info->xProxy.startFFT ();
}

#endif
