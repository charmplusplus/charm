      module tracemod
      implicit none
      external fpup_int
      interface
        subroutine ftraceBegin()
        end subroutine
        subroutine ftraceEnd()
        end subroutine
        subroutine ftraceregisteruserevent(str, ein, eout)
          character(*), intent(in) :: str
          integer, intent(in) :: ein
          integer, intent(out) :: eout
        end subroutine
        subroutine ftraceuserbracketevent(ein, bt, et)
          integer, intent(in) :: ein
          double precision, intent(in) :: bt, et
        end subroutine
      end interface
      end module



