#ifndef _NEIGHBORLBMSG_H_
#define _NEIGHBORLBMSG_H_

class NLBStatsMsg;

// this actually is not a real Charm++ message and CLBStatsMsg is just
// a regular class with pup defined.
class CkMarshalledNLBStatsMessage {
  NLBStatsMsg *msg;
  void operator=(const CkMarshalledNLBStatsMessage &) = delete;
  void operator=(CkMarshalledNLBStatsMessage &&) = delete;
  CkMarshalledNLBStatsMessage(const CkMarshalledNLBStatsMessage &rhs) = delete;
public:
  inline CkMarshalledNLBStatsMessage(void) {msg=NULL;}
  CkMarshalledNLBStatsMessage(NLBStatsMsg *m) {msg=m;} //Takes ownership of message
  CkMarshalledNLBStatsMessage(CkMarshalledNLBStatsMessage &&rhs) { msg = rhs.msg; rhs.msg = nullptr; }
  ~CkMarshalledNLBStatsMessage();
  NLBStatsMsg *getMessage(void) {void *ret=msg; msg=NULL; return (NLBStatsMsg*)ret;}
  void pup(PUP::er &p);
};
PUPmarshall(CkMarshalledNLBStatsMessage)

#endif
