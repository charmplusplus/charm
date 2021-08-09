#include "cksectionmanager.h"

void SectionMulticastMsg::pup(PUP::er &p)
{
  p | sid;
  p | ep;
  p | secmgr;
}

ck::SectionID CkSectionManager::createSectionID()
{

  return ck::SectionID(this->myPe, lastCounter++);
}

void CkSectionManager::contribute(ck::SectionID sid, int size, void *data,
                                  CkReduction::reducerType reduction, CkCallback cb
                                  )

{
  // CkMcastBaseMsg msg = new CkMcastBaseMsg();
  if(sections.find(sid) == sections.end())
      {
        ckout << "It's not created yet!" << endl;
      }
  auto &cookie = sections[sid]._cookie;
  this->mcastMgr->contribute(size, data, reduction, cookie, cb);
}

void CkSectionManager::multicast(ck::SectionID sid, int ep)
{

  int rootPE = sid.getHome();
  thisProxy[rootPE].doMulticast(sid, ep);
  // CkMcastBaseMsg *msg = new CkMcastBaseMsg(ep, );

}

void CkSectionManager::doMulticast(ck::SectionID sid, int ep)
{
  ckout << "Starting multicast on PE " << CkMyPe() << endl;
  auto &sectionID = sections[sid];
  SectionMulticastMsg *msg = new SectionMulticastMsg(sid, ep, thisProxy);
  this->mcastMgr->ArraySectionSend(NULL, ep, msg, 1, &sectionID, 0);

}


void CkSectionManager::initSectionLocal(SectionMulticastMsg *m)
{
  auto sid = m->sid;
  CkSectionInfo newCookie;
  CkGetSectionInfo(newCookie, m);
  CkSectionID newID{};
  newID._cookie = newCookie;

  if(sections.find(sid) == sections.end())
    {
      // this->mcastMgr->initDelegateMgr(newSectionInfo, aid);
      sections[sid] = newID;
    }

}

#include "cksec.def.h"
