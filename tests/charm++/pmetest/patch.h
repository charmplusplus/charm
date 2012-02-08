
#ifndef  __PATCH_H__
#define  __PATCH_H__

#include "charm++.h"
#include "pencilfft/pencil_api.h"

#define __TEST_PME_VERBOSE__   0

struct PatchInfo {
  int  my_x, my_y, my_z;
  int  nx, ny, nz;
  
  LineFFTInfo   fftinfo;

  CkCallback    cb_start;
  CkCallback    cb_done;
  
  int           niterations;
};

PUPbytes(PatchInfo)

#include "testpme.decl.h"

class Patch : public CBase_Patch {
  PatchInfo      _info;       //The info for this patch element
  LineFFTGrid  * _gridList;   //List of grid messages to send
  
  int            _nGridMsgs;     //number of grid messges to send
  int            _nReceived;     //number of grid messages received
  int            _iteration;     //number of iterations finshed
  
 public:
  
  Patch ()  {
    memset (&_info, 0, sizeof (_info));
    _nReceived = _nGridMsgs = 0;
    _iteration = 0;
    _gridList = NULL;
  }
  
  Patch (CkMigrateMessage *m) {}
  
  void startTimeStep ();
  void initialize (PatchInfo &info);
  void receiveGrid (LineFFTGridMsg *msg);
};


#define INTERPOLATION_SIZE  4   //atoms interpolate to a 4x4x4 grid

#endif
