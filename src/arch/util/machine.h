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


#define MACHINE_DEBUG 0
#if MACHINE_DEBUG
/**Controls amount of debug messages: 1 (the lowest priority) is 
extremely verbose, 2 shows most procedure entrance/exits, 
3 shows most communication, and 5 only shows rare or unexpected items.
Displaying lower priority messages doesn't stop higher priority ones.
*/
#define MACHINE_DEBUG_PRIO 2
#define MACHINE_DEBUG_LOG 1 /**Controls whether output goes to log file*/

FILE *debugLog;
# define MACHSTATE_I(prio,args) if ((debugLog)&&(prio)>=MACHINE_DEBUG_PRIO) {\
	fprintf args ; fflush(debugLog); }
# define MACHSTATE(prio,str) \
	MACHSTATE_I(prio,(debugLog,"[%.6f]> "str"\n",CmiWallTimer()))
# define MACHSTATE1(prio,str,a) \
	MACHSTATE_I(prio,(debugLog,"[%.6f]> "str"\n",CmiWallTimer(),a))
# define MACHSTATE2(prio,str,a,b) \
	MACHSTATE_I(prio,(debugLog,"[%.6f]> "str"\n",CmiWallTimer(),a,b))
# define MACHSTATE3(prio,str,a,b,c) \
	MACHSTATE_I(prio,(debugLog,"[%.6f]> "str"\n",CmiWallTimer(),a,b,c))
# define MACHSTATE4(prio,str,a,b,c,d) \
	MACHSTATE_I(prio,(debugLog,"[%.6f]> "str"\n",CmiWallTimer(),a,b,c,d))
#else
# define MACHINE_DEBUG_LOG 0
# define MACHSTATE(n,x) /*empty*/
# define MACHSTATE1(n,x,a) /*empty*/
# define MACHSTATE2(n,x,a,b) /*empty*/
# define MACHSTATE3(n,x,a,b,c) /*empty*/
# define MACHSTATE4(n,x,a,b,c,d) /*empty*/
#endif


#endif
