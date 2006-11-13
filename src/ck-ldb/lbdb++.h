#include "LBDBManager.h"

inline int LDCollectingStats(LDHandle _db)
{
  LBDB *const db = (LBDB*)(_db.handle);
  return db->StatsOn();
}
