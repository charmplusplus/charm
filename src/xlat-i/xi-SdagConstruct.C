#include "xi-SdagConstruct.h"

namespace xi {

extern void RemoveSdagComments(char *);

SdagConstruct::SdagConstruct(EToken t, const char *str)
    : type(t), traceName(NULL), con1(0), con2(0), con3(0), con4(0), elist(0)
{ text = new XStr(str); constructs = new std::list<SdagConstruct*>(); }
                                         

SdagConstruct::SdagConstruct(EToken t)
    : type(t), traceName(NULL), con1(0), con2(0), con3(0), con4(0), elist(0)
{ constructs = new std::list<SdagConstruct*>(); }

SdagConstruct::SdagConstruct(EToken t, XStr *txt)
    : type(t), traceName(NULL), text(txt), con1(0), con2(0), con3(0), con4(0), elist(0)
{ constructs = new std::list<SdagConstruct*>();  }


/***************** WhenConstruct **************/
WhenConstruct::WhenConstruct(EntryList *el, SdagConstruct *body)
: SdagConstruct(SWHEN, 0, 0, 0,0,0, body, el)
, speculativeState(0)
{ }


/***************** AtomicConstruct **************/
AtomicConstruct::AtomicConstruct(const char *code, const char *trace_name)
: SdagConstruct(SATOMIC, NULL, 0, 0, 0, 0, 0, 0)
{
  char *tmp = strdup(code);
  RemoveSdagComments(tmp);
  text = new XStr(tmp);
  free(tmp);

  if (trace_name)
  {
    tmp = strdup(trace_name);
    tmp[strlen(tmp)-1]=0;
    traceName = new XStr(tmp+1);
    free(tmp);
  }
}

}   // namespace xi
