#ifndef _NEWTYPES_H
#define _NEWTYPES_H

typedef struct {
  int srcpe;
  int ImplType;
  int ImplIndex;
  int SwitchVal;
  int NumMembers;
  CmiGroup grp;
} comID;


typedef struct {
  int msgsize;
  void *msg;
} msgstruct ;


#endif
