#include "xi-SdagConstruct.h"

namespace xi {

SdagConstruct::SdagConstruct(EToken t, const char *str)
{ init(t); text = new XStr(str); }
                                         
SdagConstruct::SdagConstruct(EToken t)
{ init(t); }

SdagConstruct::SdagConstruct(EToken t, XStr *txt)
{ init(t); }

}   // namespace xi
