#ifndef ARRAYMAP_H
#define ARRAYMAP_H

#include "arraydefs.h"
#include "ArrayMap.top.h"

class Array1D;

class ArrayMapCreateMessage : public comm_object
{
public:
  int numElements;
  ChareIDType arrayID;
  GroupIDType groupID;
};

class ArrayMap : public groupmember
{
public:
  virtual int procNum(int element) = 0;

protected:
  ArrayMap(ArrayMapCreateMessage *msg);
  void finishConstruction(void);

  ChareIDType arrayChareID;
  GroupIDType arrayGroupID;
  Array1D *array;
  int numElements;
};


#endif // ARRAYMAP_H
