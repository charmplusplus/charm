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
      end interface
