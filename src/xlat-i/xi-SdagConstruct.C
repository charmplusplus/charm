#include "xi-SdagConstruct.h"

namespace xi {

SdagConstruct::SdagConstruct(EToken t, const char *str)
{ init(t); text = new XStr(str); }
                                         
SdagConstruct::SdagConstruct(EToken t)
{ init(t); }

SdagConstruct::SdagConstruct(EToken t, XStr *txt)
{ init(t); }

  void SdagConstruct::setEntry(Entry *e) {
    entry = e;
    if (con1)
      con1->setEntry(e);
    if (con2)
      con2->setEntry(e);
    if (con3)
      con3->setEntry(e);
    if (con4)
      con4->setEntry(e);

    if (constructs)
      for (std::list<SdagConstruct *>::iterator child = constructs->begin();
           child != constructs->end();
           ++child)
        (*child)->setEntry(e);
  }

}   // namespace xi
