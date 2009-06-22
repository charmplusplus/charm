/**
   @addtogroup ComlibCharmStrategy
   @{
   
   @file 
 
*/

#ifndef ONE_TIME_MULTICAST_STRATEGY
#define ONE_TIME_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

/**
   The simplest multicast strategy. This strategy extracts the array section information, and packs the section information and the user message into a single message. The original message is delivered locally, and the new message is sent using CmiSyncListSendAndFree to all other processors containing destination objects.

   This strategy is simpler than those which are derived from the MulticastStrategy class because those maintain a persistant record of previous array section information extracted from the messages, and those provide multiple implementations of the multicast tree (such as ring or multiring or all to all). Those strategies ought to be used when multiple multicasts are sent to the same array section. If an array section is not reused, then this strategy ought to be used.

   The local messages are delivered through the array manager using the CharmStrategy::deliverToIndices methods. If a destination chare is remote, the array manager will forward it on to the pe that contains the chare.
   
*/
class OneTimeMulticastStrategy: public Strategy, public CharmStrategy {
 private:

  ComlibSectionInfo sinfo; // This is used to create the multicast messages themselves
  
  void remoteMulticast(ComlibMulticastMsg *cmsg);
  void localMulticast(CharmMessageHolder *cmsg);
  
 public:

 OneTimeMulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
    
  OneTimeMulticastStrategy();
  ~OneTimeMulticastStrategy();

  void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);}
  void insertMessage(CharmMessageHolder *msg);

  ///Called by the converse handler function
  void handleMessage(void *msg);    

  void pup(PUP::er &p);

  PUPable_decl(OneTimeMulticastStrategy);

};
#endif

/*@}*/
