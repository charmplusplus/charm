/* pg.p
 * Charm Process Groups Library
 *
 * This program comes under a limited warranty.  If you think that it may
 * have a legitimate bug, send a description of the behavior, and a copy
 * of the program to:
 *    PROCESS GROUPS CONSUMER WARRANTIES
 *    C/O Rob Neely
 *    rneely@llnl.gov
 * The offeror of this warranty has the right to ignore any requests
 *
 * Rob Neely, 1994
 */

#include "pg.int"

module PG {

extern double pow() ;
#include "pglib.h"

/*************************************************************/
/* BOC :  ProcessGroups **************************************/
/*************************************************************/
BranchOffice ProcessGroups

{

    /* BOC data structures */
    int totalBocs,me,hashSize ;
    int outstandingPartitionRequests ;
    int nextAvailableGid, nextAvailableRef ;
    RootGidType *gidList[MAX_HASH_TABLE_SIZE] ;
    BufferedMessages *outOfOrderBufList, *doneBufList,
                     *mcContBufList, *mcDelBufList ;
    ControlMsgType *mcList, *syncList ;

    /*************************************************************/
    /* ENTRY:  C R E A T E ***************************************/
    /*************************************************************/

    entry Create : (message CREATE_ROOT *pgmsg)
    {
	private int countDescendants() ;
	private AddGid() ;
	private PgDefaultSpanTreeParent() ;
	private PgDefaultSpanTreeNumChildren() ;
	private PgDefaultSpanTreeChild() ;
	RootGidType *currGid ;
	int gidHashed ;
	int i,j,d=0 ;

	CkFreeMsg(pgmsg) ;
	me = CkMyPe() ;
	totalBocs = CkNumPes() ;

	/* Set the size of the hash table.  If for some reason the user
	 * has set the number of processors equal to the hash table size,
	 * this could cause extra collisions (due to the way that gids are
	 * handed out) so bump it up to the next prime number */
	hashSize = (totalBocs != HASH_TABLE_SIZE)
	    ? HASH_TABLE_SIZE : MAX_HASH_TABLE_SIZE ;

	/* Set buffered message lists to null */
	outOfOrderBufList = NULL ;
	doneBufList = NULL ;
	mcDelBufList = NULL ;
	mcContBufList = NULL ;
	mcList = NULL ;
	syncList = NULL ;
	for(i=0;i<hashSize;i++) gidList[i] = NULL ;

	currGid = (RootGidType *)CkAlloc(sizeof(RootGidType)) ;

	currGid->gidInfo.gid = 0 ;
	currGid->gidInfo.rootProc = 0 ;

	d = PrivateCall(countDescendants(me)) ;
	currGid->groupInfo.descendants = d+1 ;   /* include myself! */

	/* Default span tree info */
	currGid->groupInfo.spanParent = PrivateCall(PgDefaultSpanTreeParent(me)) ;
	currGid->groupInfo.spanNumChildren=PrivateCall(PgDefaultSpanTreeNumChildren(me)) ;
	PrivateCall(PgDefaultSpanTreeChild(me,currGid->groupInfo.spanChildren)) ;
	currGid->groupInfo.totalGroupSize = CkNumPes() ;
	currGid->groupInfo.groupRank = CkMyPe() ;
	
	/* Since we don't have any partitions yet, clear copy list */
	currGid->groupInfo.copyInfo = NULL ;

	currGid->nextGid == NULL ;

	PrivateCall(AddGid(currGid, PG_COORD)) ;

	/* Figure out what the next available GID I can give out is.  I
	   can only give GID's which are multiples of my proc num */
	if (me==0) 
	    nextAvailableGid = totalBocs ; /* I already gave out zero */
	else 
	    nextAvailableGid = me ;

	nextAvailableRef = me ;
	outstandingPartitionRequests = 0 ;
    } /* end entry Create */

    /*************************************************************/
    /* ENTRY:  N E W  M E M B E R  L A N D  A N D  P I N  ********/
    /*************************************************************/

    entry NewMemberLandAndPin : (message NEW_MEMBER *msg)
    {
	private AddGid() ;
	private BufMsgList *FindBufferedMsgs() ;
	private DeleteBufferedMsgs() ;
	int requestor,msgFrom,rootProc ;
	int copyNum, partNum, hashedGid ;
	int i, j ;
	RootGidType *currGid ;
	CopyInfoType *currCopy ;
	BufMsgList *buf ;
	DONE_MSG *bufDoneMsg ;
	GENERIC_MESSAGE *bmsg ;

	requestor = msg->requestor ;

	/* Build a gid for this processor */
	currGid = (RootGidType *)CkAlloc(sizeof(RootGidType)) ;
	currGid->gidInfo.gid = msg->groupID ;
	currGid->gidInfo.rootProc = msg->rootProc ;
	currGid->groupInfo.spanParent = msg->msgFrom ;
	currGid->groupInfo.spanNumChildren = 0 ;
	currGid->groupInfo.groupRank = msg->rank ;
	currGid->groupInfo.descendants = 1 ; /* include myself */
	currGid->groupInfo.copyInfo = NULL ;
	currGid->groupInfo.notifyEP = msg->notifyEP ;
	currGid->groupInfo.notifyBoc = msg->notifyBoc ;
	currGid->groupInfo.notifyRefNum = msg->notifyRefNum ;
	currGid->nextGid = NULL ;

  	/* Some other span tree details that have to do with how to build
	   a span tree dynamically of the same structure as the default
	   span tree */
	currGid->groupInfo.spanDetails.direction = 0 ;
	currGid->groupInfo.spanDetails.exp = 0 ;
	currGid->groupInfo.spanDetails.range = 1 ; /* MAX^exp */
	currGid->groupInfo.spanDetails.val = 0 ;

	/* Add this new gid to my list */
	PrivateCall(AddGid(currGid, msg->groupID)) ;

	/* Check for messages that may have been buffered in NewMemberFwd
	 * due to out of order arrival (arriving before the LandAndPin
	 * message had arrived */


	/* If the buffered member list for this gid came back non-null,
	 * then traverse the list, sending the buffered messages on their
	 * way (which is basically sending them back to NewMemberFwd, where
	 * they got buffered in the first place. */
	buf = PrivateCall(FindBufferedMsgs(msg->groupID, outOfOrderBufList)) ;
	if (buf) {
	    while (buf) {
		bmsg = (GENERIC_MESSAGE *)CkCopyMsg(buf->bufMsg) ;
		SendMsgBranch(NewMemberFwd, bmsg, me) ;
		buf = buf->next ;
	    }
	    PrivateCall(DeleteBufferedMsgs(msg->groupID, &outOfOrderBufList)) ;
	}

	/* Now, check to see if we've already gotten the done message
	 * at this node.  If yes, then restart the message down the
	 * span tree */
	buf = PrivateCall(FindBufferedMsgs(msg->groupID, doneBufList)) ;
	if(buf) {
	    while(buf) {
		bufDoneMsg = (DONE_MSG *)CkCopyMsg(buf->bufMsg) ;
		SendMsgBranch(MarkPartitionDone, bufDoneMsg, me) ;
		buf = buf->next ;
	    }
	    PrivateCall(DeleteBufferedMsgs(msg->groupID, &doneBufList)) ;
	}
	CkFreeMsg(msg) ;
      } /* end entry: MemberLandAndPin */
  
    /*************************************************************/
    /* ENTRY:  N E W  M E M B E R  F W D *************************/
    /*************************************************************/

    entry NewMemberFwd : (message NEW_MEMBER *msg)
    {
	private RootGidType *FindGid() ;
	private AddBufferedMsg() ;
	private BufMsgList *FindBufferedMsgs() ;
	private DeleteBufferedMsgs() ;
	int requestor,groupID,msgFrom,rootProc ;
	int numChildren, receivor, direction ;
	int i,j ;
	RootGidType *currGid ;
	BufMsgList *msgBuf, *buf ;
	DONE_MSG *bufDoneMsg ;

	requestor = msg->requestor ;
	groupID = msg->groupID ;
	msgFrom = msg->msgFrom ;
	rootProc = msg->rootProc ;

	/* find the gid record for this groupID */
	currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;

	/* if there's no Gid Info for this groupID, and we just got
	 * the message from the coordinator (-1), then we buffer, else
	 * make new slot and hold */

	if (currGid == NULL) {
	    msgBuf = (BufMsgList *)CkAlloc(sizeof(BufMsgList)) ;
	    msgBuf->bufMsg = (GENERIC_MESSAGE*)msg ;
	    msgBuf->next = NULL ;
	    PrivateCall(AddBufferedMsg(groupID, &outOfOrderBufList, msgBuf)) ;
	}

	else {

	    /* figure out where the message should go from here */
	    numChildren =
		currGid->groupInfo.spanNumChildren ;

	    if (numChildren < MAX_SPAN_CHILDREN) {

		currGid->groupInfo.spanChildren[numChildren] = requestor ;
		currGid->groupInfo.spanNumChildren ++ ;

		msg->msgFrom = me ;
		SendMsgBranch(NewMemberLandAndPin, msg, requestor) ;
		currGid->groupInfo.descendants ++ ;
	    }
	    else {
		msg->msgFrom = me ;
		direction = currGid->groupInfo.spanDetails.direction ;
		receivor = currGid->groupInfo.spanChildren[direction] ;

		SendMsgBranch(NewMemberFwd, msg, receivor) ;

		currGid->groupInfo.descendants++ ;
	    }
	    
	    /* Update the info on how to *build* the span tree */
	    currGid->groupInfo.spanDetails.val++ ;
	    if (currGid->groupInfo.spanDetails.val >= currGid->groupInfo.spanDetails.range) {
		currGid->groupInfo.spanDetails.direction =
		    (currGid->groupInfo.spanDetails.direction + 1) % MAX_SPAN_CHILDREN ;
		currGid->groupInfo.spanDetails.val = 0 ;
		if (currGid->groupInfo.spanDetails.direction == 0) {
		    currGid->groupInfo.spanDetails.exp += 1 ;
		    currGid->groupInfo.spanDetails.range =
			(int)pow((double)MAX_SPAN_CHILDREN, (double)currGid->groupInfo.spanDetails.exp) ;
		}
	    }

	    /* Now, check to see if we've already gotten the done message
	     * at this node.  If yes, then restart the message down the
	     * span tree */
	    
	    buf = PrivateCall(FindBufferedMsgs(groupID, doneBufList)) ;
	    if(buf) {
		while(buf) {
		    bufDoneMsg = (DONE_MSG *)buf->bufMsg ;
		    SendMsgBranch(MarkPartitionDone, bufDoneMsg, me) ;
		    buf= buf->next ;
		}
		PrivateCall(DeleteBufferedMsgs(groupID, &doneBufList)) ;
	    }
	}
    } /* end entry: NewMemberFwd */


    /*************************************************************/
    /* ENTRY:  P A R T I T I O N  A T  R O O T *******************/
    /*************************************************************/
  
    entry PartitionAtRoot : (message PARTITION_AT_ROOT *prMsg)
    {
	private RootGidType *FindGid() ;
	private CopyInfoType *FindCopy() ;
	private PartInfoType *FindPartition() ;
	private AddPartition() ;
	private AddCopy() ;
	int groupID, copyNum, partNum, requestor, refNum ;
	int newGroupID, toProc, i, np ;
	EntryPointType notifyEP ;
	ChareNumType notifyBoc ;
	RootGidType *currGid ;
	CopyInfoType *currCopy ;
	PartInfoType *currPart ;
	NEW_MEMBER *nmMsg ;
	DONE_MSG *doneMsg ;

	groupID = prMsg->groupID ;
	copyNum = prMsg->copyNum ;
	partNum = prMsg->partNum ;
	requestor = prMsg->requestor ;
	refNum = prMsg->refNum ;
	notifyEP = prMsg->retEP ;
	notifyBoc = prMsg->retBoc ;
	CkFreeMsg(prMsg) ;

	/* find the gid record for this groupID */
	currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;
	if(currGid == NULL) {
	    CkPrintf("Error, called back gid %d in PartAtRoot\n",groupID) ;
	}
	else {
	    currCopy = (CopyInfoType *)PrivateCall(FindCopy(currGid->groupInfo.copyInfo, copyNum)) ;
	    
	    /* See if I'm the first one to join this copy */
	    if (currCopy == NULL) {
		currCopy = (CopyInfoType *)CkAlloc(sizeof(CopyInfoType)) ;
		currCopy->copyNum = copyNum ;
		currCopy->joinedThisCopy = 0 ;
		currCopy->needToJoinThisCopy = currGid->groupInfo.descendants ;
		currCopy->partInfo = NULL ;
		currCopy->nextCopy = NULL ;

		PrivateCall(AddCopy(&(currGid->groupInfo.copyInfo), currCopy)) ;
	    }
	    currCopy->joinedThisCopy ++ ;

	    /* Check for an error (more joining than allowed) */
	    if (currCopy->joinedThisCopy > currCopy->needToJoinThisCopy) {
		CkPrintf("ERROR in partition phase of PG lib\n") ;
		CkPrintf("Somebody must've joined more than once.\n") ;
		CkExit() ;
		return ;
	    }
	    
	    /* See if I'm the first to join this partition.  If so, then
	       make the requestor the root of this group.  Otherwise, send
	       the message down the tree via the root */
	    
	    currPart = (PartInfoType *)PrivateCall(FindPartition(currCopy->partInfo, partNum)) ;

	    if(currPart == NULL) {
		/* alloc space for this partition */
		currPart = (PartInfoType *)CkAlloc(sizeof(PartInfoType)) ;
		currPart->partNum = partNum ;
		currPart->joinedThisPartition = 1 ;
		currPart->gidInfo.rootProc = requestor ;
		currPart->gidInfo.gid = nextAvailableGid ;
		currPart->nextPart = NULL ;

		PrivateCall(AddPartition(&(currCopy->partInfo), currPart)) ;
		
		/* Build message and send to root */
		newGroupID = nextAvailableGid ;
		nmMsg = (NEW_MEMBER *)CkAllocMsg(NEW_MEMBER) ;
		nmMsg->requestor = requestor ;
		nmMsg->groupID = newGroupID ;
		nextAvailableGid += totalBocs ;
		nmMsg->msgFrom = -1 ;
		nmMsg->rank = (currPart->joinedThisPartition)-1 ;
		nmMsg->rootProc = requestor ;
		nmMsg->descendantCount=currGid->groupInfo.descendants ;
		nmMsg->notifyEP = notifyEP ;
		nmMsg->notifyBoc = notifyBoc ;
		nmMsg->notifyRefNum = refNum ;

		SendMsgBranch(NewMemberLandAndPin, nmMsg, requestor) ;
	    }
	    else {
		/* somebody has already been here with the same request */

		newGroupID = currPart->gidInfo.gid ;
		toProc = currPart->gidInfo.rootProc ;
		currPart->joinedThisPartition ++ ;
		nmMsg = (NEW_MEMBER *)CkAllocMsg(NEW_MEMBER) ;
		nmMsg->requestor = requestor ;
		nmMsg->groupID = newGroupID ;
		nmMsg->msgFrom = -1 ;
		nmMsg->rootProc = toProc ;
		nmMsg->rank = (currPart->joinedThisPartition)-1 ;
		nmMsg->descendantCount = -1 ; /* shouldn't be used */
		nmMsg->notifyEP = notifyEP ;
		nmMsg->notifyBoc = notifyBoc ;
		nmMsg->notifyRefNum = refNum ;

		SendMsgBranch(NewMemberFwd, nmMsg, toProc) ;
	    }

	    /* check to see if everyone has joined this copy */
	    if (currCopy->joinedThisCopy == currCopy->needToJoinThisCopy) {
		
		/* notify everybody - send message to root of each group,
		   who will then notify all their descendants */

		currPart=currCopy->partInfo ;
		while(currPart) {
		    toProc = currPart->gidInfo.rootProc ;

		    doneMsg = (DONE_MSG *)CkAllocMsg(DONE_MSG) ;
		    doneMsg->gid = groupID ;
		    doneMsg->newGid = currPart->gidInfo.gid ;
		    doneMsg->descendantCount = currPart->joinedThisPartition ;
		    doneMsg->totalGroupSize = currPart->joinedThisPartition ;

		    SendMsgBranch(MarkPartitionDone, doneMsg, toProc) ;

		    currPart = currPart->nextPart ;
		}
	    }
	}
    } /* end entry: PartitionAtRoot */


    /*************************************************************/
    /* ENTRY:  M A R K  P A R T I T I O N  D O N E ***************/
    /*************************************************************/
  
    entry MarkPartitionDone : (message DONE_MSG *doneMsg)
    {
	private RootGidType *FindGid() ;
	private computeBranchWeights() ;
	private AddBufferedMsg() ;
	private BufMsgList *FindBufferedMsgs() ;
	private DeleteBufferedMsgs() ;
	int doneGid, newGid, gcount ;
	int i, msgCount, receivor, descendantCount, dcount, notifyRefNum ;
	int weightList[MAX_SPAN_CHILDREN] ;
	EntryPointType notifyEp ;
	ChareNumType notifyBoc ;
	RootGidType *currGid ;
	BufMsgList *doneBuf ;
	BufMsgList *buf ;
	GENERIC_MESSAGE *mcContBufMsg, *mcDeliverBufMsg ;
	PARTITION_CREATED *pcMsg ;
	DONE_MSG *newDoneMsg ;

	newGid = doneMsg->newGid ;
	doneGid = doneMsg->gid ;
	descendantCount = doneMsg->descendantCount ;
	gcount = doneMsg->totalGroupSize ;

	/* find the gid record for this groupID */
	currGid = (RootGidType *)PrivateCall(FindGid(newGid)) ;

	if(currGid == NULL) {
	    doneBuf = (BufMsgList *)CkAlloc(sizeof(BufMsgList)) ;
	    doneBuf->bufMsg = (GENERIC_MESSAGE*)doneMsg ;
	    doneBuf->next = NULL ;
	    PrivateCall(AddBufferedMsg(newGid, &doneBufList, doneBuf)) ;
	}

	else {
	    currGid->groupInfo.totalGroupSize = gcount ;
	    notifyEp = currGid->groupInfo.notifyEP ;
	    notifyBoc = currGid->groupInfo.notifyBoc ;
	    notifyRefNum = currGid->groupInfo.notifyRefNum ;
	
	    /* If we don't even have info for this node yet, or if we don't
	       have all the descendants, then compute how many descendants 
	       there are left to nofify, and stop forwarding messageges */

	    if (currGid->groupInfo.descendants != descendantCount){
		
		/* I haven't gotten all of my descendants passing through
		 * just yet - so I'm gonna stop sending done messages down
		 * the tree, and let both NewMember functions check to
		 * see when the process should be started up again. */

		doneBuf = (BufMsgList *)CkAlloc(sizeof(BufMsgList)) ;
		doneBuf->bufMsg = (GENERIC_MESSAGE*)doneMsg ;
		doneBuf->next = NULL ;
		PrivateCall(AddBufferedMsg(newGid, &doneBufList, doneBuf)) ;
	    }
	    else {

		CkFreeMsg(doneMsg) ;
		
		/* Notify the user program of the partition being done */
		outstandingPartitionRequests-- ;
		pcMsg = (PARTITION_CREATED *)CkAllocMsg(PARTITION_CREATED) ;
		pcMsg->newGid = newGid ;
		SetRefNumber(pcMsg, notifyRefNum) ;
		SendMsgBranch(notifyEp, pcMsg, me, notifyBoc) ;
		
		/* forward "done message" to the span children */
		dcount = currGid->groupInfo.descendants-1 ; /* take me out */
		PrivateCall(computeBranchWeights(dcount, weightList)) ;
		
		msgCount = currGid->groupInfo.spanNumChildren ;
		
		for(i=0 ; i<msgCount ; i++) {
		    newDoneMsg = (DONE_MSG *)CkAllocMsg(DONE_MSG) ;
		    newDoneMsg->gid = doneGid ;
		    newDoneMsg->newGid = newGid ;
		    newDoneMsg->descendantCount = weightList[i] ;
		    newDoneMsg->totalGroupSize = gcount ;
		    
		    receivor = currGid->groupInfo.spanChildren[i] ;

		    SendMsgBranch(MarkPartitionDone, newDoneMsg, receivor) ;
		    
		}

		/* Check to see if we have any buffered multicast messages */
		buf = PrivateCall(FindBufferedMsgs(newGid, mcContBufList)) ;
		if(buf) {
		    while(buf) {
			mcContBufMsg=(GENERIC_MESSAGE*)CkCopyMsg(buf->bufMsg);
			SendMsgBranch(MulticastControl, mcContBufMsg, me) ;
			buf = buf->next ;
		    }
		    PrivateCall(DeleteBufferedMsgs(newGid, &mcContBufList)) ;
		}
		buf = PrivateCall(FindBufferedMsgs(newGid, mcDelBufList)) ;
		if(buf) {
		    while(buf) {
  		        mcDeliverBufMsg=
			    (GENERIC_MESSAGE*)CkCopyMsg(buf->bufMsg) ; 
			SendMsgBranch(MulticastDeliver, mcDeliverBufMsg, me) ;
			buf = buf->next ;
		    }
		    PrivateCall(DeleteBufferedMsgs(newGid, &doneBufList)) ;
		}
	    }
	}
    }  /* end entry MarkPartionDone */
    


    /*************************************************************/
    /* ENTRY:  M U L T I C A S T  C O N T R O L ******************/
    /*************************************************************/
  
    entry MulticastControl : (message MULTICAST_MESSAGE *mmsg)
    {
	private RootGidType *FindGid() ;
	private AddControlMsg() ;
	private ControlMsgType *FindControlMsg() ;
	private DeleteControlMsg() ;
	private AddBufferedMsg() ;
	int controlRefNum, userRefNum, gid ;
	int i, msgCount, receivor ;
	RootGidType *currGid ;
	EntryPointType deliverEP ;
	ChareNumType deliverBoc ;
	GENERIC_MESSAGE *gmsg ;
	MULTICAST_MESSAGE *fwdMsg ;
	ControlMsgType *mcInfo ;
	BufMsgList *mcControlBuf ;
  
 	/* find the gid record for this groupID */
	controlRefNum = mmsg->controlRefNum ;

	mcInfo = PrivateCall(FindControlMsg(controlRefNum,mcList)) ;

	if(mcInfo == NULL) {
	    mcInfo = (ControlMsgType *)CkAlloc(sizeof(ControlMsgType)) ;
	    mcInfo->controlRefNum = controlRefNum ;
	    mcInfo->gotMessage = 0 ;
	    mcInfo->next = NULL ;
	    PrivateCall(AddControlMsg(mcInfo, &mcList)) ;
	}

	mcInfo->gotControl = 1 ;
	mcInfo->groupID = mmsg->gid ;
	mcInfo->controlRefNum = controlRefNum ;
	mcInfo->userRef = mmsg->userRefNum ;
	mcInfo->deliverEP = mmsg->retEP ;
	mcInfo->deliverBoc = mmsg->retBoc ;
	mcInfo->next = NULL ;
	
	/* forward this message down the span tree */
	gid = mcInfo->groupID ;
	currGid = (RootGidType *)PrivateCall(FindGid(gid)) ;
	if (currGid == NULL) {
	    /* A multicast message has arrived from another possible
	     * group member before I got word from my root group member. */
	    if(outstandingPartitionRequests == 0) {
		/* Since I'm not expecting any oustanding partition requests
		 * back, and if I were legally in this group, I would have
		 * to have made a request (since the root group member sent
		 * a done message), this must be an error. */
		CkPrintf("** Error in Multicast.  Invalid gid %d at proc %d\n",
			 gid, me) ;
		CkExit() ;
	    }
	    else {
		  /* Buffer the message, and when the done message comes,
		   * unbuffer it and send it */
		  mcControlBuf = (BufMsgList *)CkAlloc(sizeof(BufMsgList)) ;
		  mcControlBuf->bufMsg = gmsg ;
		  mcControlBuf->next = NULL ;
		  PrivateCall(AddBufferedMsg(gid, &mcContBufList,
					     mcControlBuf)) ;
	    }
	}
	else {
	    msgCount = currGid->groupInfo.spanNumChildren ;
	    for(i=0 ; i < msgCount ; i++) {
		
		/* alloc and build message to be forwarded */
		fwdMsg = (MULTICAST_MESSAGE *)CkCopyMsg(mmsg) ;
		receivor = currGid->groupInfo.spanChildren[i] ;

		SendMsgBranch(MulticastControl, fwdMsg, receivor) ;
	    }
	    /* Check to see if we have this message buffered already.
	       If yes, then deliver it and forward it down the span tree */
	    if (mcInfo->gotMessage == 1) {
		gmsg = mcInfo->controlBufMsg ;
		SendMsgBranch(MulticastDeliver, gmsg, me) ;
	    }
	}
    } /* end entry multicastControl */
    

    /*************************************************************/
    /* ENTRY:  M U L T I C A S T  D E L I V E R ******************/
    /*************************************************************/
  
    entry MulticastDeliver : (message GENERIC_MESSAGE *gmsg)
    {
	private RootGidType *FindGid() ;
	private AddControlMsg() ;
	private ControlMsgType *FindControlMsg() ;
	private DeleteControlMsg() ;
	private AddBufferedMsg() ;
	private BufMsgList *FindBufferedMsgs() ;
	private DeleteBufferedMsgs() ;
	int controlRefNum, i, msgCount, gid, receivor ;
	RootGidType *currGid ;
	EntryPointType deliverEP ;
	ChareNumType deliverBoc ;
	GENERIC_MESSAGE *fwdMsg ;
	ControlMsgType *mcInfo ;
	BufMsgList *mcDeliverBuf ;
	
 	controlRefNum = GetRefNumber(gmsg) ;
	mcInfo = PrivateCall(FindControlMsg(controlRefNum,mcList)) ;

	if(mcInfo == NULL) {
	    mcInfo = (ControlMsgType *)CkAlloc(sizeof(ControlMsgType)) ;
	    mcInfo->controlRefNum = controlRefNum ;
	    mcInfo->gotControl = 0 ;
	    mcInfo->next = NULL ;
	    mcInfo->gotMessage = 1 ;
	    mcInfo->controlBufMsg = gmsg ;
	    PrivateCall(AddControlMsg(mcInfo, &mcList)) ;
	}
	else {

	  gid = mcInfo->groupID ;
	  currGid = (RootGidType *)PrivateCall(FindGid(gid)) ;
	  if (currGid == NULL) {
	      /* A multicast message has arrived from another possible
	       * group member before I got word from my root group member. */
	      
	      if(outstandingPartitionRequests == 0) {
		  /* Since I'm not expecting any oustanding partition requests
		   * back, and if I were legally in this group, I would have
		   * to have made a request (since the root group member sent
		   * a done message), this must be an error. */
		  CkPrintf("** Error in Multicast.Invalid gid %d at proc %d\n",
			   gid, me) ;
		  CkExit() ;
	      }
	      else {
		  /* Buffer the message, and when the done message comes,
		   * unbuffer it and send it */
		  mcDeliverBuf = (BufMsgList *)CkAlloc(sizeof(BufMsgList)) ;
		  mcDeliverBuf->bufMsg = gmsg ;
		  mcDeliverBuf->next = NULL ;
		  PrivateCall(AddBufferedMsg(gid, &mcDelBufList,
					     mcDeliverBuf)) ;
	      }
	  }
	  else {
	    msgCount = currGid->groupInfo.spanNumChildren ;

	    for(i=0 ; i < msgCount ; i++) {
		fwdMsg = (GENERIC_MESSAGE *)CkCopyMsg(gmsg) ;
		receivor = currGid->groupInfo.spanChildren[i] ;
		SendMsgBranch(MulticastDeliver, fwdMsg, receivor) ;
	    }
	    /* If we've gotten the control message already, then deliver it */
	    if (mcInfo->gotControl == 1) {
		deliverEP = mcInfo->deliverEP ;
		deliverBoc = mcInfo->deliverBoc ;
		SetRefNumber(gmsg, mcInfo->userRef) ;
		SendMsgBranch(deliverEP, gmsg, me, deliverBoc) ;
		PrivateCall(DeleteControlMsg(controlRefNum,&mcList)) ;
	      }
	  }
	}
    }


    /*************************************************************/
    /* ENTRY:  Synchronize support functions *********************/
    /*************************************************************/

    entry SynchronizeUpTree : (message SYNCHRONIZE_MESSAGE *inMsg)
    {
	private ControlMsgType *FindControlMsg() ;
	private AddControlMsg() ;
	private RootGidType *FindGid() ;
	RootGidType *currGid ;
	ControlMsgType *syncInfo ;
	int groupID, key, myParent ;

	groupID = inMsg->gid ;
	key = inMsg->controlRefNum ;
	SetRefNumber(inMsg, key) ;  /* To quiet purify */

	/* find the gid record for this groupID */
	currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;

	if(currGid == NULL) {
	    /* Unlike the multicast messages, a synchronize message
	     * coming up the tree CANNOT arrive at a node before that
	     * node has received it's partition_done message.  Since
	     * this message is coming up the span tree, it means that
	     * the span tree must be defined above this point, since
	     * it's the partition_done message which tells the final
	     * node who it's parent is.  Get it? */
	    CkPrintf("** Error in Synchronize. Can't find gid %d on proc %d\n",
		     groupID, me) ;
	    CkExit() ;
	}

	syncInfo = PrivateCall(FindControlMsg(key, syncList)) ;
	if(syncInfo == NULL) {
	    syncInfo = (ControlMsgType *)CkAlloc(sizeof(ControlMsgType)) ;
	    syncInfo->controlRefNum = key ;
	    syncInfo->count = 0 ;
	    syncInfo->next = NULL ;
 	    PrivateCall(AddControlMsg(syncInfo, &syncList)) ;
	}

	syncInfo->count++ ;

	if (syncInfo->count == currGid->groupInfo.spanNumChildren+1) {
	    /* We got all the messages from our children */
	    if (currGid->groupInfo.spanParent == -1) {
		/* This is the root - multicast the done statement */
		BranchCall(MyBocNum(),Multicast_libcall(groupID, inMsg, SynchronizeDeliver, MyBocNum())) ;

	    }
	    else {
		/* Send on up to my parents */
		myParent = currGid->groupInfo.spanParent ;
		SendMsgBranch(SynchronizeUpTree, inMsg, myParent) ;
	    }
	}
	else 
	    CkFreeMsg(inMsg) ;
    }


    entry SynchronizeDeliver : (message SYNCHRONIZE_MESSAGE *smsg)
    {
	private ControlMsgType *FindControlMsg() ;
	private DeleteControlMsg() ;
	int key, gid ;
	ControlMsgType *syncInfo ;
	EntryPointType ep ;
	ChareIDType id ;
	GENERIC_MESSAGE *gmsg ;

	key = smsg->controlRefNum ;
	gid = smsg->gid ;
	CkFreeMsg(smsg) ;

	syncInfo = PrivateCall(FindControlMsg(key, syncList)) ;

	if(syncInfo == NULL) {
	    CkPrintf("Error in SynchronizeDeliver.  Can't find key %d\n",key) ;
	}
	else {
	    gmsg = syncInfo->controlBufMsg ;
	    ep = syncInfo->deliverEP ;
	    id = syncInfo->deliverId ;
	    SendMsg(ep, gmsg, id) ;
	}
	PrivateCall(DeleteControlMsg(key, &syncList)) ;
    }

/*********************************************************************/
/*** THE FOLLOWING ROUTINES ARE ALL PRIVATE FUNCTIONS TO THE *********/
/*** PROCESS GROUP LIBRARY, AND ARE USED IN MAINTAINING DATA *********/
/*** STRUCTURES   ****************************************************/
/*********************************************************************/

/* Note: the Add/Find/Delete Buffered Msg functions are "overloaded"
 * to work with both types of messages which can be buffered (either
 * new member requests, or done messages.  This is done by using the
 * GENERIC_MESSAGE type, and requiring the user to pass in the pointer
 * to the head of the list
 */

/*********************************************************************/
/*** Private function: Add/Find/Delete BufferedMsg *******************/
/*********************************************************************/

private AddBufferedMsg(groupID, head, new)
int groupID ;
BufferedMessages **head ;
BufMsgList *new ;    
{
    private BufMsgList *FindBufferedMsgs() ;
    BufMsgList *groupHead, *temp1 ;
    BufferedMessages *temp, *scan ;

    groupHead = PrivateCall(FindBufferedMsgs(groupID, *head)) ;
    if (groupHead) {
	temp1 = groupHead ;
	while(temp1->next) temp1=temp1->next ;
	temp1->next = new ;
    }
    else {
	temp = (BufferedMessages *)CkAlloc(sizeof(BufferedMessages)) ;
	temp->groupID = groupID ;
	temp->bufMsgHead = new ;
	temp->next = NULL ;
	if(*head == NULL) 
	    *head = temp ;
	else {
	    scan = *head ;
	    while(scan->next) scan=scan->next ;
	    scan->next = temp ;
	}
    }
}

private BufMsgList *FindBufferedMsgs(groupID, head)
int groupID ;
BufferedMessages *head ;
{
    BufferedMessages *temp1 ;
    if(head == NULL) 
	return(NULL) ;
    else {
	temp1 = head ;
	while (temp1) {
	    if(temp1->groupID == groupID) return(temp1->bufMsgHead) ;
	    temp1=temp1->next ;
	}
	return(NULL) ;
    }
}

private DeleteBufferedMsgs(groupID, head)
int groupID ;
BufferedMessages **head ;
{
    BufMsgList *groupHead ;
    BufferedMessages *temp, *skip ;
    
    groupHead = PrivateCall(FindBufferedMsgs(groupID, *head)) ;
    if (groupHead) {
	temp = *head ;
	if(temp->bufMsgHead == groupHead) 
	    *head = (*head)->next ;
	else {
	    while(temp->next->bufMsgHead != groupHead) temp=temp->next ;
	    if(temp->next == NULL) 
		skip = NULL ;
	    else 
		skip = temp->next->next ;
	    CkFree(temp->next->bufMsgHead) ; 
	    temp->next = skip ;
	}
    }
    else {
	CkPrintf("%d: ERROR!!!!  Trying to delete a buffered message list\n") ;
	CkPrintf("  which doesn't exist!\n") ;
    }
}

/*********************************************************************/
/*** Private function: Add/Find/Delete ControlMsg *********************/
/*********************************************************************/

private AddControlMsg(ptr, head)
ControlMsgType *ptr ;
ControlMsgType **head ;
{
    ControlMsgType *temp ;
    if(*head == NULL) 
	*head = ptr ;
    else {
	temp = *head ;
	while(temp->next) temp=temp->next ;
	temp->next = ptr ;
    }
}

private ControlMsgType *FindControlMsg(ref, head)
int ref ;
ControlMsgType *head ;
{
    ControlMsgType *temp ;
    temp = head ;
    while (temp != NULL) {
	if(temp->controlRefNum == ref) 
	    return(temp) ;
	else 
	    temp=temp->next ;
    }
    return(NULL) ;
}

private DeleteControlMsg(ref, head)
int ref ;
ControlMsgType **head ;
{
    ControlMsgType *temp, *skip ;
    temp = *head ;
    if(temp->controlRefNum == ref) {
	*head = temp->next ;
	CkFree(temp) ;
    }
    else {
	while(temp->next->controlRefNum != ref) temp=temp->next ;
	if(temp->next == NULL) 
	    skip = NULL ;
	else 
	    skip = temp->next->next ;
	CkFree(temp->next) ;
	temp->next = skip ;
    }
}

/*********************************************************************/
/*** Private function: Add/Find Gid **********************************/
/*********************************************************************/

private HashGid(gid)
int gid ;
{
    return (gid % hashSize) ;
}

private AddGid(gidPtr, gid)
RootGidType *gidPtr ;
int gid ;
{
    RootGidType *temp ;
    int hashedGid ;
    hashedGid = PrivateCall(HashGid(gid)) ;
    if(gidList[hashedGid] == NULL)
	gidList[hashedGid] = gidPtr ;
    else {
	temp = gidList[hashedGid] ;
	while(temp->nextGid) temp=temp->nextGid ;
	temp->nextGid = gidPtr ;
    }
}

private RootGidType *FindGid(groupID)
int groupID ;
{
    RootGidType *temp ;
    int hashedGid ;
    hashedGid = PrivateCall(HashGid(groupID)) ;
    temp = gidList[hashedGid] ;
    while (temp != NULL) {
	if(temp->gidInfo.gid == groupID)
	    return(temp) ;
	else 
	    temp=temp->nextGid ;
    }
    return(NULL) ;
}

/*********************************************************************/
/*** Private function: Add/Find Copy *********************************/
/*********************************************************************/

private AddCopy(head, new)
CopyInfoType **head, *new ;
{
    CopyInfoType *temp ;
    if(*head == NULL) 
	*head = new ;
    else {
	temp = *head ;
	while(temp->nextCopy) temp=temp->nextCopy ;
	temp->nextCopy = new ;
    }
}
    
private CopyInfoType *FindCopy(head, copyNum)
CopyInfoType *head ;
int copyNum ;
{
    CopyInfoType *temp ;
    if(head == NULL) 
	return(NULL) ;
    else {
	temp = head ;
	while(temp != NULL) {
	    if(temp->copyNum == copyNum)
		return(temp) ;
	    else 
		temp=temp->nextCopy ;
	}
	return(NULL) ;
    }
}

/*********************************************************************/
/*** Private function: Add/Find Partition ****************************/
/*********************************************************************/

private AddPartition(head, new)
PartInfoType **head, *new ;
{
    PartInfoType *temp ;
    if(*head == NULL)
	*head = new ;
    else {
	temp = *head ;
	while(temp->nextPart) temp=temp->nextPart ;
	temp->nextPart = new ;
    }
}

private PartInfoType *FindPartition(head, partNum)
PartInfoType *head ;
int partNum ;
{
    PartInfoType *temp ;
    if(head == NULL) 
	return(NULL) ;
    else {
	temp = head ;
	while(temp != NULL) {
	    if(temp->partNum == partNum)
		return(temp) ;
	    else 
		temp=temp->nextPart ;
	}
	return(NULL) ;
    }
}

/*************************************************************************/
/******** Functions for building the default span tree *******************/
/*************************************************************************/
/* I do not use the system library, since the PG library depends heavily */
/* upon the implementation of the default spanning tree to build its     */
/* span trees dynamically - most importantly the number of children at   */
/* each node.  At the time of this writing, I stole the routines below   */
/* from the system library.                                              */
/*************************************************************************/
private int PgDefaultSpanTreeParent(node)
int node;
{
    if (node == 0)
         return -1;
    else 
	 return ( ((node - 1) / MAX_SPAN_CHILDREN) );   /* integer division */
}

private PgDefaultSpanTreeChild(node, children)
int node, *children;
{
    int i;

    for (i = 1; i <= MAX_SPAN_CHILDREN ; i++)
	if (MAX_SPAN_CHILDREN * node + i < CkNumPes())
	     children[i-1] = node * MAX_SPAN_CHILDREN + i;
	else children[i-1] = -1;
}

private int PgDefaultSpanTreeNumChildren(node)
int node;
{
    if ((node + 1) * MAX_SPAN_CHILDREN < CkNumPes())
         return(MAX_SPAN_CHILDREN);
    else if (node * MAX_SPAN_CHILDREN + 1 >= CkNumPes())
	 return 0;
    else return ((CkNumPes() - 1) - node * MAX_SPAN_CHILDREN);
}

/*********************************************************************/
/*** Private function: computeBranchWeights **************************/
/*********************************************************************/


private computeBranchWeights(dc, list)
int dc ;
int list[MAX_SPAN_CHILDREN] ;
{
    int i, val, pos, n, exp ;
    pos = exp = n = 0 ;
    val = 1 ;
    for(i=0;i<MAX_SPAN_CHILDREN;i++) list[i]=0 ;    
    while( (n+val) <= dc) {
	list[pos] += val ;
	n += val ;
	pos++ ;
	if(pos >= MAX_SPAN_CHILDREN) {
	    pos = 0 ;
	    exp++ ;
	    val = (int)pow((double)MAX_SPAN_CHILDREN, (double)exp) ;
	}
    }
    list[pos] += (dc - n) ; 
}

	    
/*********************************************************************/
/*** Private function: countDescendants ******************************/
/*********************************************************************/

private int countDescendants(nodeNum)
int nodeNum ;
/* This will only work for group 0 (root/mother) */
{
    private PgDefaultSpanTreeNumChildren() ;
    private PgDefaultSpanTreeChild() ;
    private countDescendants() ;
    int children[MAX_SPAN_CHILDREN] ;
    int i, ndx, total ;
    ndx = PrivateCall(PgDefaultSpanTreeNumChildren(nodeNum)) ;
    total = ndx ;
    PrivateCall(PgDefaultSpanTreeChild(nodeNum, children)) ;
    for(i=0 ; i<ndx ; i++) 
	total += PrivateCall(countDescendants(children[i])) ;
    return(total) ;
}



/*********************************************************************/
/**** SEMI-PUBLIC Group span tree accessor functions *****************/
/*********************************************************************/

public int PgMySpanTreeParent_libcall(groupID)
int groupID ;
{
    RootGidType *currGid ;
    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;
    /* Lookup the root for the group we are partitioning */
    if (currGid == NULL) {
	CkPrintf("******Error in PgMySpanTreeParent! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else {
	return currGid->groupInfo.spanParent ;
    }
}

public int PgMyNumSpanTreeChildren_libcall(groupID)
int groupID ;
{
    RootGidType *currGid ;
    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;

    /* Lookup the root for the group we are partitioning */
    if (currGid == NULL) {
	CkPrintf("******Error in PgMyNumSpanTreeChildren! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else {
	return currGid->groupInfo.spanNumChildren ;
    }
}

public PgMySpanTreeChildren_libcall(groupID, children)
int groupID ;
int *children ;
{
    int i ;
    RootGidType *currGid ;
    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;

    /* Lookup the root for the group we are partitioning */
    if (currGid == NULL) {
	CkPrintf("******Error in PgMySpanTreeChildren! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else { 
	for(i=0;i<currGid->groupInfo.spanNumChildren ; i++) {
	    children[i] = currGid->groupInfo.spanChildren[i] ;
	}
    }
}


/*************************************************************/
/* SEMI-PUBLIC PgGroupSize_libcall **********************/
/*************************************************************/

public int PgGroupSize_libcall(groupID)
int groupID ;
{
    RootGidType *currGid ;
    int rootProc ;

    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;
    if (currGid == NULL) {
	CkPrintf("*******Error in PgGroupSize! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else {
	return (currGid->groupInfo.totalGroupSize) ;
    }
}

/*************************************************************/
/* SEMI-PUBLIC PgMyRank_libcall **********************/
/*************************************************************/

public int PgMyRank_libcall(groupID)
int groupID ;
{
    RootGidType *currGid ;
    int rootProc ;

    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;
    if (currGid == NULL) {
	CkPrintf("*******Error in PgMyRank! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else {
	return (currGid->groupInfo.groupRank) ;
    }
}

/*********************************************************************/
/**** SEMI-PUBLIC Multicast_libcall ******************************/
/*********************************************************************/

public Multicast_libcall(groupID, gmsg, returnEP, returnBoc)
int groupID ;
GENERIC_MESSAGE *gmsg ;
EntryPointType returnEP ;
ChareNumType returnBoc ;
{
    int rootProc ;
    MULTICAST_MESSAGE *mmsg ;
    RootGidType *currGid ;

    /* find the gid record for this groupID */
    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;
    
    /* Lookup the root for the group we are partitioning */
    if (currGid == NULL) {
	CkPrintf("******Error in multicast! Couldn't find group ID.\n") ;
	CkExit() ;
    }

    else {
	rootProc = currGid->gidInfo.rootProc ;
    
	/* start building the new message */
	mmsg = (MULTICAST_MESSAGE *)CkAllocMsg(MULTICAST_MESSAGE) ;
	
	/* Store the users reference number in case they were using it */
	mmsg->gid = groupID ;
	mmsg->userRefNum = GetRefNumber(gmsg) ;  
	mmsg->controlRefNum = nextAvailableRef ;
	mmsg->retEP = returnEP ;
	mmsg->retBoc = returnBoc ;
	
	/* Send off both the control and user message */
	SendMsgBranch(MulticastControl, mmsg, rootProc) ;

 	/* Fill in the reference number on the users message and send it off */
	SetRefNumber(gmsg, nextAvailableRef) ; 
	SendMsgBranch(MulticastDeliver, gmsg, rootProc) ;
	
	nextAvailableRef += totalBocs ;
    }
} /* end public function :: multicast */



/*************************************************************/
/* SEMI-PUBLIC Partition_libcall **************************/
/*************************************************************/

public Partition_libcall(groupID, copyNum, partNum, ref, retEP, retBoc)
int groupID ;
int copyNum ;
int partNum ;
int ref ;
EntryPointType retEP ;
ChareNumType retBoc ;
{
    int rootProc ;
    PARTITION_AT_ROOT *partMsg ;
    RootGidType *currGid ;
    me = CkMyPe() ;

    outstandingPartitionRequests++ ;
    
    /* find the gid record for this groupID */
    currGid = (RootGidType *)PrivateCall(FindGid(groupID)) ;

    /* Lookup the root for the group we are partitioning */
    if (currGid == NULL) {
	CkPrintf("******Error in partition! Couldn't find group ID.\n") ;
	CkExit() ;
    }
    else {
	rootProc = currGid->gidInfo.rootProc ;
    
	/* Build message and send off to the root of this partition */
	partMsg = (PARTITION_AT_ROOT *)CkAllocMsg(PARTITION_AT_ROOT) ;
	partMsg->groupID = groupID ;
	partMsg->copyNum = copyNum ;
	partMsg->partNum = partNum ;
	partMsg->requestor = me ;
	partMsg->refNum = ref ;
	partMsg->retEP = retEP ;
	partMsg->retBoc = retBoc ;
	
	SendMsgBranch(PartitionAtRoot, partMsg, rootProc) ;
    }
    
}  /* end public function :: partition */


/*************************************************************/
/* SEMI-PUBLIC Synchronize_libcall ************************/
/*************************************************************/

public Synchronize_libcall(groupID, key, gmsg, ep, id) 
int groupID ;
int key ;
GENERIC_MESSAGE *gmsg ;
EntryPointType ep ;
ChareIDType id ;
{
    private AddControlMsg() ;
    private ControlMsgType *FindControlMsg() ;
    SYNCHRONIZE_MESSAGE *smsg ;
    RootGidType *currGid ;
    ControlMsgType *syncInfo ;

    /* Buffer the message until we get word to send it */
    syncInfo = PrivateCall(FindControlMsg(key, syncList)) ;
	
    if(syncInfo == NULL) {
	syncInfo = (ControlMsgType *)CkAlloc(sizeof(ControlMsgType)) ;
	syncInfo->controlRefNum = key ;
	syncInfo->count = 0 ;
	syncInfo->next = NULL ;
	PrivateCall(AddControlMsg(syncInfo, &syncList)) ;
    }
    /* This data should only/always be filled in by the local processor */
    syncInfo->deliverEP = ep ;
    syncInfo->deliverId = id ;
    syncInfo->controlBufMsg = gmsg ;
    
    /* Deposit my info to the local branch */
    smsg = (SYNCHRONIZE_MESSAGE*)CkAllocMsg(SYNCHRONIZE_MESSAGE) ;
    smsg->gid = groupID ;
    smsg->controlRefNum = key ;
    SendMsgBranch(SynchronizeUpTree, smsg, me) ;	
}


} /* end boc::ProcessGroups */



/*************************************************************/
/* CHARE :  PGInitHandler ************************************/
/*************************************************************/
chare PGInitHandler {

    EntryPointType ReturnEP ;
    ChareNumType ReturnBOC ;

    entry Start : (message PG_CREATE_ROOT *pgmsg) {

	ChareIDType myID ;

	ReturnBOC = pgmsg->boc ;
	ReturnEP = pgmsg->ep ;
	MyChareID(&myID) ;
	
	CreateBoc(ProcessGroups,ProcessGroups@Create,pgmsg,Distribute,myID) ;
    }

    entry Distribute : (message ROOT_GROUP_CREATED *inMsg) {

	BroadcastMsgBranch(ReturnEP, inMsg, ReturnBOC) ;
	ChareExit() ;
    }

}

/*********************************************************************/
/*** Public function(s): createRootGroup ********************************/
/*********************************************************************/

/* Use this function if creating the root group outside of CharmInit. */
void CreateRootGroupMsg(ReturnEP, ReturnBOC) 
ChareNumType ReturnBOC ;
EntryPointType ReturnEP ;
{
    PG_CREATE_ROOT *pgmsg ;
    
    /* this function is called from a single processor, and will create
       a root group for the calling BOC, and return the boc num and root
       GID to the ReturnBOC and ReturnEP specified in the parameters */
    pgmsg = (PG_CREATE_ROOT *) CkAllocMsg(PG_CREATE_ROOT) ;
    pgmsg->ep = ReturnEP ;
    pgmsg->boc = ReturnBOC ;
    CreateChare(PGInitHandler, PGInitHandler@Start, pgmsg) ;
}

/*********************************************************************/

/* Use this function if creating the group from inside CharmInit */
ChareNumType CreateRootGroup()
{
    PG_CREATE_ROOT *pgmsg ;
    pgmsg = (PG_CREATE_ROOT *) CkAllocMsg(PG_CREATE_ROOT) ;
    return CreateBoc(ProcessGroups,ProcessGroups@Create,pgmsg) ;
}

/*************************************************************/
/* PUBLIC FUNCTION: P A R T I T I O N ************************/
/*************************************************************/

void Partition(coordBoc, groupID, copyNum, partNum, ref, retEP, retBoc)
ChareNumType coordBoc ;
int groupID ;
int copyNum ;
int partNum ;
int ref ;
EntryPointType retEP ;
ChareNumType retBoc ;
{
    BranchCall(coordBoc, ProcessGroups@Partition_libcall(groupID, copyNum, partNum, ref, retEP, retBoc)) ;
}


/*********************************************************************/
/**** Public function: multicast **************************************/
/*********************************************************************/

void Multicast(coordBoc, groupID, gmsg, returnEP, returnBoc)
ChareNumType coordBoc ;
int groupID ;
GENERIC_MESSAGE *gmsg ;
EntryPointType returnEP ;
ChareNumType returnBoc ;
{
    BranchCall(coordBoc,ProcessGroups@Multicast_libcall(groupID,gmsg,returnEP,returnBoc)) ;    

}

/*********************************************************************/
/**** Public function: synchronize ***********************************/
/*********************************************************************/

/* NOTE: key must be uniuque across group ID's.  That is, the library
 * will not distinguish between key=10 if two different groups use it */

void Synchronize(coordBoc, groupID, key, gmsg, ep, id)
ChareNumType coordBoc ;
int groupID ;
int key ;
GENERIC_MESSAGE *gmsg ;
EntryPointType ep ;
ChareIDType id ;
{
   BranchCall(coordBoc,ProcessGroups@Synchronize_libcall(groupID,key,gmsg,ep,id)) ;
}

/*********************************************************************/
/**** Public functions: Span tree creators and accesors **************/
/*********************************************************************/

int PgMySpanTreeParent(coordBoc, groupID)
ChareNumType coordBoc ;
int groupID ;
{
    return BranchCall(coordBoc,ProcessGroups@PgMySpanTreeParent_libcall(groupID));
}

/*********************************************************************/

int PgMyNumSpanTreeChildren(coordBoc, groupID)
ChareNumType coordBoc ;
int groupID ;
{
    return BranchCall(coordBoc,ProcessGroups@PgMyNumSpanTreeChildren_libcall(groupID)) ; 
}

/*********************************************************************/

void PgMySpanTreeChildren(coordBoc, groupID, children)
ChareNumType coordBoc ;
int groupID ;
int *children ;
{
    BranchCall(coordBoc, ProcessGroups@PgMySpanTreeChildren_libcall(groupID, children)) ;
}

/*********************************************************************/

int PgGroupSize(coordBoc, groupID)
ChareNumType coordBoc ;
int groupID ;
{
    return BranchCall(coordBoc,ProcessGroups@PgGroupSize_libcall(groupID)) ;
}

/*********************************************************************/

int PgMyRank(coordBoc, groupID)
ChareNumType coordBoc ;
int groupID ;
{
    return BranchCall(coordBoc,ProcessGroups@PgMyRank_libcall(groupID)) ;
}

} /* end module::PG */
