#include "oocAPI.h"
#include <list>

class CkOOCManager:public CBase_CkOOCManager{
  int seq;
  bool dirCreated;
  public:
    CkOOCManager();
    int addHandler(CkIOMetaData * md);
    void scheduleTask(OOCTask * task);
    void printAnalysis();
};
