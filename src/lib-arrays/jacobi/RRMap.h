#ifndef RRMAP_H
#define RRMAP_H

#include "arraydefs.h"
#include "RRMap.top.h"
#include "ArrayMap.h"

class RRMap : public ArrayMap
{
public:
  RRMap(ArrayMapCreateMessage *msg);
  ~RRMap(void);

  int procNum(int element);
};


#endif // RRMAP_H
