#ifndef ARRAYELEMENT_H
#define ARRAYELEMENT_H

#include "arraydefs.h"
#include "ArrayElement.top.h"

class Array1D;
class ArrayElementCreateMessage;
class ArrayElementMigrateMessage;
class ArrayElementExitMessage;

class ArrayElement : public groupmember
{
friend class Array1D;
public:
  ArrayElement(ArrayElementCreateMessage *msg);
  ArrayElement(ArrayElementMigrateMessage *msg);
  void finishConstruction(void);
  void migrate(int where);
  void finishMigration(void);
  void exit(ArrayElementExitMessage *msg);

protected:
  virtual int packsize(void);
  virtual void pack(void *pack);

  ChareIDType arrayChareID;
  GroupIDType arrayGroupID;
  Array1D *thisArray;
  int numElements;
  int thisIndex;
};

class ArrayElementCreateMessage : public comm_object
{
public:
  int numElements;
  ChareIDType arrayID;
  GroupIDType groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementMigrateMessage : public comm_object
{
public:
  int numElements;
  ChareIDType arrayID;
  GroupIDType groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementExitMessage : public comm_object
{
public:
  int dummy;
};

#endif // ARRAYELEMENT_H
