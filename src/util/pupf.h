      external fpup_int
      external fpup_ints
      external fpup_char
      external fpup_chars
      external fpup_short
      external fpup_shorts
      external fpup_real
      external fpup_reals
      external fpup_double
      external fpup_doubles
      external fpup_logical
      external fpup_logicals
      interface
        function fpup_issizing(p)
          INTEGER :: p
          logical fpup_issizing
        end function
        function fpup_ispacking(p)
          INTEGER :: p
          logical fpup_ispacking
        end function
        function fpup_isunpacking(p)
          INTEGER :: p
          logical fpup_isunpacking
        end function
        function fpup_isdeleting(p)
          INTEGER :: p
          logical fpup_isdeleting
        end function
        function fpup_isuserlevel(p)
          INTEGER :: p
          logical fpup_isuserlevel
        end function
        subroutine fpup_complex(p,c)
          INTEGER p
          complex c
        end subroutine
        subroutine fpup_complexes(p,c,size)
          INTEGER p
          complex,pointer,dimension(:) :: c
          INTEGER size
        end subroutine
        subroutine fpup_doublecomplex(p,c)
          INTEGER p
          double complex c
        end subroutine
        subroutine fpup_doublecomplexes(p,c,size)
          INTEGER p
          double complex,pointer,dimension(:) :: c
          INTEGER size
        end subroutine
      end interface
