      module tracemod
      implicit none
      interface
        subroutine ftraceBegin()
        end subroutine
        subroutine ftraceEnd()
        end subroutine
        subroutine ftraceRegisterUserevent(str, ein, eout)
          character(*), intent(in) :: str
          integer, intent(in) :: ein
          integer, intent(out) :: eout
        end subroutine
        subroutine ftraceUserBracketEvent(ev, bt, et)
          integer, intent(in) :: ev 
          double precision, intent(in) :: bt, et
        end subroutine
        subroutine ftraceUserEvent(ev)
          integer, intent(in) :: ev 
        end subroutine
        subroutine ftraceFlushLog()
        end subroutine
        subroutine ftraceRegisterFunc(str, outidx)
          character(*), intent(in) :: str
          integer, intent(in) :: outidx
        end subroutine
        subroutine ftraceBeginFunc(idx)
          integer, intent(in) :: idx
        end subroutine
        subroutine ftraceEndFunc(idx)
          integer, intent(in) :: idx
        end subroutine
        subroutine ftracePhaseEnd()
        end subroutine
      end interface
      end module

