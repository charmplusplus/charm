#ifndef _ARRAYRING_H
#define _ARRAYRING_H

#include "arrayring.decl.h"
#include "megatest.h"

class arrayMessage : public CMessage_arrayMessage {
  char data[7];
public:
  int iter;
  arrayMessage(void) { sprintf(data, "Array!"); iter = 0; };
  int check(void) { return !strcmp(data, "Array!"); }
};

class arrayRing_array : public CBase_arrayRing_array
{
 public:
  arrayRing_array();
  arrayRing_array(CkMigrateMessage *msg) {}
  void start(arrayMessage *msg);
};

#endif
