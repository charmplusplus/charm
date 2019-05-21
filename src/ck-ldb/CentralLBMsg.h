#ifndef _CENTRALLBMSG_H_
#define _CENTRALLBMSG_H_

#include <vector>

class CLBStatsMsg;

// this actually is not a real Charm++ message and CLBStatsMsg is just
// a regular class with pup defined.
class CkMarshalledCLBStatsMessage {
  std::vector<CLBStatsMsg *> msgs;
  void operator=(const CkMarshalledCLBStatsMessage &) = delete;
  void operator=(CkMarshalledCLBStatsMessage &&) = delete;
  CkMarshalledCLBStatsMessage(const CkMarshalledCLBStatsMessage&) = delete;
public:
  CkMarshalledCLBStatsMessage(void) {}
  CkMarshalledCLBStatsMessage(CLBStatsMsg *m) { add(m); } //Takes ownership of message
  CkMarshalledCLBStatsMessage(CkMarshalledCLBStatsMessage &&rhs) : msgs(std::move(rhs.msgs)) {}
  ~CkMarshalledCLBStatsMessage() { free(); }
  void add(CLBStatsMsg *m) { if (m!=NULL) msgs.push_back(m); } 
  void add(CkMarshalledCLBStatsMessage &&msg);     // add multiple messages
  CLBStatsMsg *getMessage(int c=0) {CLBStatsMsg *ret=msgs[c]; msgs[c]=NULL; return ret;}
  int  getCount() { return msgs.size(); }
  void pup(PUP::er &p);
  void free();
};
PUPmarshall(CkMarshalledCLBStatsMessage)

#endif
