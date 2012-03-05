/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "converse.h"

#include "LBProfit.h"

// a naive formular to determining if profitable
// if Lmax - Lavg less than a threshold, then skip LB
int LBProfit::profitable(BaseLB::ProcStats *procArray, int np)
{
  int doit = 1;
  double lmax = 0.0;
  double ltotal = 0.0;
  for (int i=0; i<np; i++) {
    BaseLB::ProcStats &p = procArray[i];
    // FIXME
    double objTime = p.total_walltime - p.idletime - p.bg_walltime;
    if (objTime > lmax) lmax = objTime;
    ltotal += objTime;
  }
  double lavg = ltotal/np;
  if ((lmax - lavg) / lavg < 0.01) doit = 0;

  return doit;
}

/*@}*/
