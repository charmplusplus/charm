#ifndef _CENTRALLBMSG_H_
#define _CENTRALLBMSG_H_

class CLBStatsMsg;

// this actually is not a real Charm++ message and CLBStatsMsg is just
// a regular class with pup defined.
class CkMarshalledCLBStatsMessage {
  CkVec<CLBStatsMsg *> msgs;
  //Don't use these: only pass by reference
  void operator=(const CkMarshalledCLBStatsMessage &);
public:
  CkMarshalledCLBStatsMessage(void) {}
  CkMarshalledCLBStatsMessage(CLBStatsMsg *m) { add(m); } //Takes ownership of message
  CkMarshalledCLBStatsMessage(const CkMarshalledCLBStatsMessage &);
  ~CkMarshalledCLBStatsMessage() { free(); }
  void add(CLBStatsMsg *m) { if (m!=NULL) msgs.push_back(m); } 
  void add(CkMarshalledCLBStatsMessage &msg);     // add multiple messages
  CLBStatsMsg *getMessage(int c=0) {CLBStatsMsg *ret=msgs[c]; msgs[c]=NULL; return ret;}
  int  getCount() { return msgs.size(); }
  void pup(PUP::er &p);
  void free();
};
PUPmarshall(CkMarshalledCLBStatsMessage)

#endif
