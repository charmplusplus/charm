
#ifndef  __PENCIL_API_H__
#define  __PENCIL_API_H__

#include "pencilfft.h"

inline void configureLineFFTInfo (LineFFTInfo *info,
				  int sizeX,  int sizeY,  int sizeZ,
				  int grainX, int grainY, int grainZ,
				  CkCallback  *kcallback,
				  LineFFTCompletion complid,
				  bool              normalize,
				  int               numiter) 
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
  info->numIter      = numiter;
}


inline void  createLineFFTArray (LineFFTInfo *info) {

  CkAssert (info->sizeX > 0);
  CkAssert (info->sizeY > 0);
  CkAssert (info->sizeZ > 0);
  
  CkAssert (info->grainX > 0);
  CkAssert (info->grainY > 0);
  CkAssert (info->grainZ > 0);
  
  int nx, ny, nz;

  nx = info->sizeX / info->grainX;
  ny = info->sizeY / info->grainY;
  nz = info->sizeZ / info->grainZ;

  printf ("Creating Line FFT Array (%dx%dx%d, %dx%dx%d) on %d nodes, %d PEs\n",
	  info->sizeX, 
	  info->sizeY, 
	  info->sizeZ, 
	  info->grainX, 
	  info->grainY, 
	  info->grainZ,
	  CmiNumNodes(),
	  CkNumPes());

  info->mapx =  CProxy_PencilMapX::ckNew(*info);
  info->mapy =  CProxy_PencilMapY::ckNew(*info);
  info->mapz =  CProxy_PencilMapZ::ckNew(*info);
  
  CkArrayOptions optsx;
  optsx.setMap (info->mapx);
  info->xProxy = CProxy_LineFFTArray::ckNew(*info, 
					    (int) PHASE_X, optsx);

  CkArrayOptions optsy;
  optsy.setMap (info->mapy);
  info->yProxy = CProxy_LineFFTArray::ckNew(*info, 
					    (int) PHASE_Y, optsy);

  CkArrayOptions optsz;
  optsz.setMap (info->mapz);
  info->zProxy = CProxy_LineFFTArray::ckNew(*info, 
					    (int) PHASE_Z, optsz);
  
  int x,y,z;
  for (z = 0; z < (info->sizeZ)/(info->grainZ); z ++) 
    for (y = 0; y < (info->sizeY)/(info->grainY); y++)
      info->xProxy(y, z).insert(*info, (int) PHASE_X);
  
  info->xProxy.doneInserting();
  
  for (x = 0; x < (info->sizeX)/(info->grainX); x ++)
    for (z = 0; z < (info->sizeZ)/(info->grainZ); z ++) 
      info->yProxy(z, x).insert(*info, (int) PHASE_Y);
  
  info->yProxy.doneInserting();
  
  for (y = 0; y < (info->sizeY)/(info->grainY); y ++) 
    for (x = 0; x < (info->sizeX)/(info->grainX); x ++) 
      info->zProxy(x, y).insert(*info, (int) PHASE_Z);
  
  info->zProxy.doneInserting();
}


inline void startLineFFTArray (LineFFTInfo *info) {
  info->xProxy.startFFT ();
}

#endif
