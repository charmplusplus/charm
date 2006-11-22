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
