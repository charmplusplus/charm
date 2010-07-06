/**
 * \defgroup Machine Converse Machine Layer
 * \brief Common structures for all implementations of the machine layer
 * \ingroup Converse

 The machine layer of Converse consist of few common files to all architectures, shown in this module, which are:
 - immediate.c
 - machine-smp.c
 - machine-smp.h
 - machine.h
 - pcqueue.h
 - persist-comm.c
 - persist_impl.h

 These files describe the common characteristics of all implementations, and provide converse, and every language built on top of it the same functional interface in all machines, however different they are.

 In addition to these common files, there are files called "machine.c" which are the real implementation for the different architectures. Every file is in a different directory, and get selected for compilation at compile time through the "build" script. With this implementation, only one single machine layer can be compiled into the runtime system. Changing architecture needs a different compilation.
*/

/** @file
 * common machine header
 * @ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

#ifndef MACHINE_H
#define MACHINE_H

/* Extra error checking for comm and immediate flag-locks */
#if 0 && CMK_SHARED_VARS_UNAVAILABLE
#  define MACHLOCK_DEBUG
#  define MACHLOCK_ASSERT(l,str) \
	if (!(l)) \
		CmiAbort("Lock assertation failed: " __FILE__ " " str);
#else /* no extra flag/lock checking, or SMP version */
#  define MACHLOCK_ASSERT(l,str) /* empty */
#endif
/** Be warned all ye who turn this on .. 
   Turning MACHINE_DEBUG on can lead to problems like strange 
	 hangs because of horible stuff like printfs inside SIGIO */

#define MACHINE_DEBUG 0
#if MACHINE_DEBUG
/**Controls amount of debug messages: 1 (the lowest priority) is 
extremely verbose, 2 shows most procedure entrance/exits, 
3 shows most communication, and 5 only shows rare or unexpected items.
Displaying lower priority messages doesn't stop higher priority ones.
*/
#define MACHINE_DEBUG_PRIO 3
#define MACHINE_DEBUG_LOG 1 /**Controls whether output goes to log file*/

extern FILE *debugLog;
# define MACHSTATE_I(prio,args) if ((debugLog)&&(prio)>=MACHINE_DEBUG_PRIO) {\
	CmiMemLock(); fprintf args ; fflush(debugLog); CmiMemUnlock(); }
# define MACHSTATE(prio,str) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer()))
# define MACHSTATE1(prio,str,a) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer(),a))
# define MACHSTATE2(prio,str,a,b) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer(),a,b))
# define MACHSTATE3(prio,str,a,b,c) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer(),a,b,c))
# define MACHSTATE4(prio,str,a,b,c,d) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer(),a,b,c,d))
# define MACHSTATE5(prio,str,a,b,c,d,e) \
	MACHSTATE_I(prio,(debugLog,"[%d %.6f]> "str"\n",CmiMyRank(),CmiWallTimer(),a,b,c,d,e))
#else
# define MACHINE_DEBUG_LOG 0
# define MACHSTATE(n,x) /*empty*/
# define MACHSTATE1(n,x,a) /*empty*/
# define MACHSTATE2(n,x,a,b) /*empty*/
# define MACHSTATE3(n,x,a,b,c) /*empty*/
# define MACHSTATE4(n,x,a,b,c,d) /*empty*/
# define MACHSTATE5(n,x,a,b,c,d,e) /*empty*/
#endif


#define COMM_SERVER_FROM_SMP             0
#define COMM_SERVER_FROM_INTERRUPT       1
#define COMM_SERVER_FROM_WORKER          2

#endif

/*@}*/
