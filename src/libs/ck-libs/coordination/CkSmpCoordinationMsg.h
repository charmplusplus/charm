#ifndef CK_SMP_COORDINATION_MSG_H
#define CK_SMP_COORDINATION_MSG_H

#include "CkSmpCoordination.decl.h"
#include "ckmulticast.h"

struct CkSmpCoordinationMsg : public CkMcastBaseMsg, public CMessage_CkSmpCoordinationMsg {
};

struct CkSmpCoordinationLeaderMsg : public CkMcastBaseMsg, public CMessage_CkSmpCoordinationLeaderMsg {
  int leaderPe;

  CkSmpCoordinationLeaderMsg(int leader){
    leaderPe = leader;
  }
};

template<typename PayloadType>
struct CkSmpCoordinationPayloadMsg : public CkMcastBaseMsg, public CMessage_CkSmpCoordinationPayloadMsg<PayloadType> {
  PayloadType payload;

  CkSmpCoordinationPayloadMsg(const PayloadType &pld) : 
    payload(pld)
  {}
};

#endif // CK_SMP_COORDINATION_MSG_H
