#ifndef _CENTRALLBMSG_H_
#define _CENTRALLBMSG_H_

class CLBStatsMsg;

// this actually is not a real Charm++ message and CLBStatsMsg is just
// a regular class with pup defined.
class CkMarshalledCLBStatsMessage {
  CLBStatsMsg *msg;
  //Don't use these: only pass by reference
  void operator=(const CkMarshalledCLBStatsMessage &);
public:
  CkMarshalledCLBStatsMessage(void) {msg=NULL;}
  CkMarshalledCLBStatsMessage(CLBStatsMsg *m) {msg=m;} //Takes ownership of message
  CkMarshalledCLBStatsMessage(const CkMarshalledCLBStatsMessage &);
  ~CkMarshalledCLBStatsMessage();
  CLBStatsMsg *getMessage(void) {void *ret=msg; msg=NULL; return (CLBStatsMsg*)ret;}
  void pup(PUP::er &p);
};
PUPmarshall(CkMarshalledCLBStatsMessage);

#endif
