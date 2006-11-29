#include "LBDBManager.h"


inline int LDCollectingStats(LDHandle _db)
{
#if CMK_LBDB_ON
  LBDB *const db = (LBDB*)(_db.handle);
  return db->StatsOn();
#else
  return 0;
#endif
}

inline int LDRunningObject(LDHandle _h, LDObjHandle* _o)
{
#if CMK_LBDB_ON
  LBDB *const db = (LBDB*)(_h.handle);

  // same as LBDatabase::RunningObject
  if (db->ObjIsRunning()) {
    *_o = db->RunningObj();
    return 1;
  } else return 0;
#else
  return 0;
#endif
}

