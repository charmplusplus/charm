/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.12  1995-09-07 05:25:48  gursoy
 * added new macros related with HANDLE_INIT_MSg handler
 *
 * Revision 2.11  1995/07/25  00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.10  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.9  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/07/19  22:15:32  jyelon
 * *** empty log message ***
 *
 * Revision 2.7  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.5  1995/07/06  13:59:12  gursoy
 * well still modifying CsvAccess(... MsgStructTable)
 *
 * Revision 2.4  1995/07/06  13:45:34  gursoy
 * removed includ trans_defl.h
 *
 * Revision 2.3  1995/07/06  04:44:55  gursoy
 * fixed MsgToStructTable definition (it is a Csv type variable and
 * secondly I included trans_defs.h to use MSG_STRUCT type because
 * Csv macros cannot use struct xxx .
 *
 * Revision 2.2  1995/06/29  21:35:23  narain
 * Declared PACKED state macros, #define for LDBFillBlock and changed
 * members of MSG_STRUCT from pack and unpack to packfn and unpackfn
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.10  1995/04/24  20:06:06  sanjeev
 * *** empty log message ***
 *
 * Revision 1.9  1995/04/24  19:52:44  sanjeev
 * Changed CkAsyncSend to Cmi
 * Changed CkAsyncSend to CmiSyncSend in CkSend()
 *
 * Revision 1.8  1995/04/23  21:02:26  sanjeev
 * Removed Core...
 *
 * Revision 1.7  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.6  1995/03/25  18:25:47  sanjeev
 * *** empty log message ***
 *
 * Revision 1.5  1995/03/24  16:42:38  sanjeev
 * *** empty log message ***
 *
 * Revision 1.4  1995/03/17  23:37:42  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1995/03/12  17:09:37  sanjeev
 * changes for new msg macros
 *
 * Revision 1.2  1994/11/11  05:31:23  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:45  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#ifndef NO_PACK
#define NO_PACK 0
#endif

#ifndef UNPACKED
#define UNPACKED 1
#endif

#ifndef PACKED 
#define PACKED 2
#endif


/* CsvExtern(struct msg_struct*, MsgToStructTable); */

#define UNPACK(envelope) if (GetEnv_isPACKED(envelope) == PACKED) \
{ \
        void *unpackedUsrMsg; \
        void *usrMsg = USER_MSG_PTR(envelope); \
        (*(CsvAccess(MsgToStructTable)[GetEnv_packid(envelope)].unpackfn)) \
                (usrMsg, &unpackedUsrMsg); \
        if (usrMsg != unpackedUsrMsg) \
        /* else unpacked in place */ \
        { \
                int temp_i; \
                int temp_size; \
                char *temp1, *temp2; \
                /* copy envelope */ \
                temp1 = (char *) envelope; \
                temp2 = (char *) ENVELOPE_UPTR(unpackedUsrMsg); \
                temp_size = (char *) usrMsg - temp1; \
                for (temp_i = 0; temp_i<temp_size; temp_i++) \
                        *temp2++ = *temp1++; \
                CmiFree(envelope); \
                envelope = ENVELOPE_UPTR(unpackedUsrMsg); \
        } \
        SetEnv_isPACKED(envelope, UNPACKED); \
}


#define PACK(env) 	if (GetEnv_isPACKED(env) == UNPACKED) \
        /* needs packing and not already packed */ \
        { \
		int size; \
        	char *usermsg, *packedmsg; \
                /* make it +ve to connote a packed msg */ \
                SetEnv_isPACKED(env, PACKED); \
                usermsg = USER_MSG_PTR(env); \
                (*(CsvAccess(MsgToStructTable)[GetEnv_packid(env)].packfn)) \
                        (usermsg, &packedmsg, &size); \
                if (usermsg != packedmsg) \
                        env = ENVELOPE_UPTR(packedmsg); \
        }\



#define CkCheck_and_Send(PE, env)\
    {\
        if (PE==CmiMyPe())\
            { ClrEnv_LdbFull(env); HANDLE_INCOMING_MSG(env); }\
        else {\
            SetEnv_LdbFull(env);\
            CldFillLdb(PE, LDB_ELEMENT_PTR(env)); \
            PACK(env); \
            CmiSetHandler(env,CsvAccess(HANDLE_INCOMING_MSG_Index)); \
            CmiSyncSend(PE,CmiSize(env),env); \
            CmiFree(env); \
        }\
    }

#define CkCheck_and_Broadcast(env) { \
        SetEnv_LdbFull(env);\
        CldFillLdb(CK_PE_ALL_BUT_ME, LDB_ELEMENT_PTR(env)); PACK(env); \
        CmiSetHandler(env,CsvAccess(HANDLE_INCOMING_MSG_Index)); \
	CmiSyncBroadcast(CmiSize(env),env); \
	CmiFree(env) ; \
        }

#define CkCheck_and_BroadcastNoFree(env) { \
        SetEnv_LdbFull(env);\
        CldFillLdb(CK_PE_ALL_BUT_ME, LDB_ELEMENT_PTR(env)); PACK(env); \
        CmiSetHandler(env,CsvAccess(HANDLE_INCOMING_MSG_Index)); \
	CmiSyncBroadcast(CmiSize(env),env); UNPACK(env);  \
        }

#define CkCheck_and_BroadcastNoFreeNoLdb(env) { \
        ClrEnv_LdbFull(env);\
        PACK(env); \
        CmiSetHandler(env,CsvAccess(HANDLE_INCOMING_MSG_Index)); \
	CmiSyncBroadcast(CmiSize(env),env); UNPACK(env);  \
        }

#define CkCheck_and_BroadcastAll(env) { \
        SetEnv_LdbFull(env);\
        CldFillLdb(CK_PE_ALL, LDB_ELEMENT_PTR(env)); PACK(env); \
        CmiSetHandler(env,CsvAccess(HANDLE_INCOMING_MSG_Index)); \
	CmiSyncBroadcastAllAndFree(CmiSize(env),env);\
        }



#define CkCheck_and_BcastInitNFNL(env) { \
        ClrEnv_LdbFull(env);\
        PACK(env); \
        CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index)); \
	CmiSyncBroadcast(CmiSize(env),env); UNPACK(env);  \
        }


#define CkCheck_and_BcastInitNL(env) { \
	ClrEnv_LdbFull(env);\
	PACK(env); \
	CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index)); \
	CmiSyncBroadcast(CmiSize(env),env); CmiFree(env);  \
	}

#endif
