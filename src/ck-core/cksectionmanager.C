#include "cksectionmanager.h"

ck::SectionID CkSectionManager::createSectionID()
{
  return ck::SectionID(CkMyPe(), lastCounter++);
}
CkSectionManager::CkSectionManager() {}

#include "cksec.def.h"
