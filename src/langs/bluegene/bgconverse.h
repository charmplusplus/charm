/*****************************************************************************
          Blue Gene Middle Layer for Converse program
*****************************************************************************/
                                                                                
#ifndef _BGCONVERSE_H_
#define _BGCONVERSE_H_

#include "blue.h"

#undef CmiRegisterHandler
#undef CmiNumberHandler
#undef CmiNumberHandlerEx
#define CmiRegisterHandler(x)        BgRegisterHandler((BgHandler)(x))
#define CmiNumberHandler(n, x)       BgNumberHandler(n, (BgHandler)(x))
#define CmiNumberHandlerEx(n, x, p)  BgNumberHandlerEx(n, (BgHandlerEx)(x), p)

#undef CmiMsgHeaderSizeBytes
#define CmiMsgHeaderSizeBytes	      CmiBlueGeneMsgHeaderSizeBytes

#undef CmiMyPe
#undef CmiNumPes
#undef CmiMyRank
#undef CmiMyNode
#undef CmiNumNodes
#undef CmiMyNodeSize

#define CmiMyPe()           (BgGetGlobalWorkerThreadID())
#define CmiNumPes()	    (BgNumNodes()*BgGetNumWorkThread())
#define CmiMyRank()	    (BgGetThreadID())
#define BgNodeRank()	    (BgMyRank()*BgGetNumWorkThread()+BgGetThreadID())
#define CmiMyNode()	    (BgMyNode())
#define CmiNumNodes()	    (BgNumNodes())
#define CmiMyNodeSize()	    (BgGetNumWorkThread())

#undef CpvDeclare
#undef CpvExtern
#undef CpvStaticDeclare
#undef CpvInitialize
#undef CpvAccess
#undef CpvAccessOther

#define CpvDeclare        BpvDeclare
#define CpvExtern         BpvExtern
#define CpvStaticDeclare  BpvStaticDeclare
#define CpvInitialize     BpvInitialize
#define CpvAccess         BpvAccess
#define CpvAccessOther    BpvAccessOther
                                                                                
#undef CsvDeclare
#undef CsvExtern
#undef CsvStaticDeclare
#undef CsvInitialize
#undef CsvAccess

#define CsvDeclare        BnvDeclare
#define CsvExtern         BnvExtern
#define CsvStaticDeclare  BnvStaticDeclare
#define CsvInitialize     BnvInitialize
#define CsvAccess         BnvAccess

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree
                                                                                
#undef CmiSyncNodeSend
#undef CmiSyncNodeSendAndFree
#undef CmiSyncNodeBroadcast
#undef CmiSyncNodeBroadcastAndFree
#undef CmiSyncNodeBroadcastAll
#undef CmiSyncNodeBroadcastAllAndFree

#define CmiSyncSend             	BgSyncSend
#define CmiSyncSendAndFree      	BgSyncSendAndFree
#define CmiSyncBroadcast        	BgSyncBroadcast
#define CmiSyncBroadcastAndFree 	BgSyncBroadcastAndFree
#define CmiSyncBroadcastAll     	BgSyncBroadcastAll
#define CmiSyncBroadcastAllAndFree      BgSyncBroadcastAllAndFree
                                                                                
#define CmiSyncNodeSend                 BgSyncNodeSend
#define CmiSyncNodeSendAndFree          BgSyncNodeSendAndFree
#define CmiSyncNodeBroadcast            BgSyncNodeBroadcast
#define CmiSyncNodeBroadcastAndFree     BgSyncNodeBroadcastAndFree
#define CmiSyncNodeBroadcastAll         BgSyncNodeBroadcastAll
#define CmiSyncNodeBroadcastAllAndFree  BgSyncNodeBroadcastAllAndFree

#undef CmiSyncListSendAndFree
#define CmiSyncListSendAndFree		BgSyncListSendAndFree

#undef CsdEnqueueLifo
#define CsdEnqueueLifo(m)  CmiSyncSendAndFree(CmiMyPe(),sizeof(m), (char*)(m));

#endif

