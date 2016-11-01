#include <converse.h>

extern "C" {

  CpvDeclare(int, cmiMyPeIdle);
#if CMK_SMP && CMK_TASKQUEUE
  CsvDeclare(unsigned int, idleThreadsCnt);
#endif

  void CsdBeginIdle(void)
  {
    CcdCallBacks();
#if CMK_TRACE_ENABLED && CMK_PROJECTOR
    _LOG_E_PROC_IDLE(); 	/* projector */
#endif
#if CMK_SMP && CMK_TASKQUEUE
    if (CpvAccess(cmiMyPeIdle) !=1) {
      CpvAccess(cmiMyPeIdle) = 1;
      CmiMemoryAtomicIncrement(CsvAccess(idleThreadsCnt));
    }
#else
    CpvAccess(cmiMyPeIdle) = 1;
#endif // CMK_SMP
    CcdRaiseCondition(CcdPROCESSOR_BEGIN_IDLE) ;
  }

  void CsdStillIdle(void)
  {
    CcdRaiseCondition(CcdPROCESSOR_STILL_IDLE);
  }

  void CsdEndIdle(void)
  {
#if CMK_TRACE_ENABLED && CMK_PROJECTOR
    _LOG_E_PROC_BUSY(); 	/* projector */
#endif
#if CMK_SMP && CMK_TASKQUEUE
    if (CpvAccess(cmiMyPeIdle) != 0){
      CpvAccess(cmiMyPeIdle) = 0;
      CmiMemoryAtomicDecrement(CsvAccess(idleThreadsCnt));
    }
#else
    CpvAccess(cmiMyPeIdle) = 0;
#endif // CMK_SMP
    CcdRaiseCondition(CcdPROCESSOR_BEGIN_BUSY) ;
  }

}
