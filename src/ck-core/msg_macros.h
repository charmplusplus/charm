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
 * Revision 2.2  1995-06-29 22:35:57  narain
 * Changed cast of LDB_ELEMENT_UPTR to (void *) from (LDB_ELEMENT_PTR *)
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.4  1995/03/17  23:37:57  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1995/03/12  17:10:05  sanjeev
 * changes for new msg macros
 *
 * Revision 1.2  1994/11/11  05:24:19  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:08  brunner
 * Initial revision
 *
 ***************************************************************************/


/**********************************************************************
 THIS FILE CONSTITUTES THE NEW FORMAT OF A Charm MESSAGE 
 This file provides access macros for extracting the different
 sections of a message. The organisation of a message is as follows 

           -------------------------------------
           | env | ldb | pad | user | priority |
           -------------------------------------
 
   The sizes of the fields are as follows:
 
       envelope      : sizeof(ENVELOPE)
                        (ENVELOPE is defined in env_macros.h)
			First word in ENVELOPE is the core language field.
 
       ldb           : LDB_ELEM_SIZE is a global variable defined by the
                        load balancing module
 
       pad           : padding to ensure that the message header ends at a
                       double word boundary.
 
       user          : the user message data.
 
       priority      : bit-vector (variable size)


************************************************************************
 The following variables reflect the message format above. If any
change is made to the format, the initialization of the variables must
be altered. The variables are initialized in InitializeMessageMacros()
in main/common.c. Compile time constants are #defines. 
All variables reflect sizes in BYTES.			
************************************************************************/
#ifndef MSG_MACROS_H
#define MSG_MACROS_H

CpvExtern(int, PAD_SIZE);
CpvExtern(int, HEADER_SIZE);
CpvExtern(int, LDB_ELEM_SIZE);

#define ENVELOPE_SIZE sizeof(ENVELOPE)

CpvExtern(int, _CK_Env_To_Usr);
#define _CK_Env_To_Ldb ENVELOPE_SIZE

CpvExtern(int, _CK_Ldb_To_Usr);
#define _CK_Ldb_To_Env (-ENVELOPE_SIZE)

CpvExtern(int, _CK_Usr_To_Env);
CpvExtern(int, _CK_Usr_To_Ldb);




#define TOTAL_MSG_SIZE(usrsize, priosize) (CpvAccess(HEADER_SIZE) + priosize + usrsize)
#define CHARRED(x) ((char *) (x))



/**********************************************************************/
/* The following macros assume that -env- is an ENVELOPE pointer */
/**********************************************************************/
#define LDB_ELEMENT_PTR(env)  \
	(void *) (CHARRED(env) + _CK_Env_To_Ldb)

#define USER_MSG_PTR(env)\
    (CHARRED(env) + CpvAccess(_CK_Env_To_Usr))

#define COPY_PRIORITY(env1, env2) {\
        if ( GetEnv_PrioType(env1) == 0 ) { \
                SetEnv_PrioType(env2,0) ; \
                SetEnv_IntegerPrio(env2, GetEnv_IntegerPrio(env1)) ; \
        } \
        else { \
                char *ptr1, *ptr2; \
                SetEnv_PrioType(env2,1) ; \
                ptr1 = (char *)env1 + *((int *)GetEnv_PrioOffset(env1));\
                ptr2 = (char *)env2 + *((int *)GetEnv_PrioOffset(env2));\
                memcpy( ((char *) ptr2), ((char *) ptr1), \
                                GetEnv_PrioSize(env1) );   \
        } \
}
 
#define MSG_PRIORITY_PTR(env, priorityptr) GetEnv_PriorityPtr(env,priorityptr)
 
#define INSERT_PRIO_OFFSET(env, usrsize, priosize)\
{\
    if ( priosize > 4 ) { \
        SetEnv_PrioType(env,1) ; \
        SetEnv_PrioOffset(env,usrsize + CpvAccess(_CK_Env_To_Usr)) ; \
        SetEnv_PrioSize(env,priosize) ; \
    } \
    else \
        SetEnv_PrioType(env,0) ; \
}



/**********************************************************************/
/* the following macros assume that -ldbptr- is a LDB_ELEMENT pointer */
/**********************************************************************/

#define ENVELOPE_LDBPTR(ldbptr) \
	(ENVELOPE *) (CHARRED(ldbptr) + _CK_Ldb_To_Env)

#define USR_MSG_LDBPTR(ldbptr) \
	(CHARRED(ldbptr) + CpvAccess(_CK_Ldb_To_Usr))


/**********************************************************************/
/* the following macros assume that "usrptr" is a pointer to a user defined 
   message */
/**********************************************************************/
#define ENVELOPE_UPTR(usrptr)\
	(ENVELOPE *) (CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Env))

#define LDB_UPTR(usrptr)\
    (LDB_ELEMENT *) (CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Ldb))

#define PRIORITY_UPTR(usrptr) \
    (PVECTOR *) ( ReturnEnv_PriorityPtr(CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Env)) )

#define MSG_PRIORITY_SIZE(usrptr) \
		GetEnv_PrioSize(CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Env))

#endif
