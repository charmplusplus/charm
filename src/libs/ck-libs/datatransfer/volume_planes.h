/**
  
*/
#ifndef __UIUC_VOLUME_LASS_H
#define __UIUC_VOLUME_LASS_H

/**
 Compute the volume bounded by this set of halfspaces (hyperplanes).
  \param planes Values of halfplanes: array of (nPlanes) (G_d+1) rationals.
       We're inside the volume if, for each row A=&planes[i*4],
          A[0] * x + A[1] * y + A[2] * z <= A[3]
  \param nPlanes Number of halfspaces (hyperplanes).
*/
double computeVolumePlanes(const double *planes,int nPlanes);

#endif
