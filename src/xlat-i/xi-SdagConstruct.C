#include "xi-SdagConstruct.h"

namespace xi {

SdagConstruct::SdagConstruct(EToken t, const char *str)
    : type(t), traceName(NULL), con1(0), con2(0), con3(0), con4(0), elist(0)
{ text = new XStr(str); constructs = new std::list<SdagConstruct*>(); }
                                         
SdagConstruct::SdagConstruct(EToken t)
    : type(t), traceName(NULL), con1(0), con2(0), con3(0), con4(0), elist(0)
{ constructs = new std::list<SdagConstruct*>(); }

SdagConstruct::SdagConstruct(EToken t, XStr *txt)
    : type(t), traceName(NULL), text(txt), con1(0), con2(0), con3(0), con4(0), elist(0)
{ constructs = new std::list<SdagConstruct*>();  }

}   // namespace xi
