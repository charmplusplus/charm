      module pupmod
      implicit none
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
      interface pup
        module procedure pi,pia1d,pia2d,pia3d,pc,pca1d,pca2d,pca3d
        module procedure ps,psa1d,psa2d,psa3d,pr,pra1d,pra2d,pra3d
        module procedure pd,pda1d,pda2d,pda3d
      end interface
      contains
      subroutine pi(p, i)
        INTEGER :: p
        integer, intent(inout) :: i
        call fpup_int(p, i)
      end subroutine
      subroutine pia1d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:) :: ia
        call fpup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia2d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:,:) :: ia
        call fpup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia3d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:,:,:) :: ia
        call fpup_ints(p, ia, size(ia))
      end subroutine
      subroutine pc(p, i)
        INTEGER :: p
        character*1, intent(inout) :: i
        call fpup_char(p, i)
      end subroutine
      subroutine pca1d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:) :: ia
        call fpup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca2d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:,:) :: ia
        call fpup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca3d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:,:,:) :: ia
        call fpup_chars(p, ia, size(ia))
      end subroutine
      subroutine ps(p, i)
        INTEGER :: p
        integer(kind=2), intent(inout) :: i
        call fpup_short(p, i)
      end subroutine
      subroutine psa1d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:) :: ia
        call fpup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa2d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:) :: ia
        call fpup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa3d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:) :: ia
        call fpup_shorts(p, ia, size(ia))
      end subroutine
      subroutine pr(p, i)
        INTEGER :: p
        real(kind=4), intent(inout) :: i
        call fpup_real(p, i)
      end subroutine
      subroutine pra1d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:) :: ia
        call fpup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra2d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:) :: ia
        call fpup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra3d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:) :: ia
        call fpup_reals(p, ia, size(ia))
      end subroutine
      subroutine pd(p, i)
        INTEGER :: p
        real(kind=8), intent(inout) :: i
        call fpup_double(p, i)
      end subroutine
      subroutine pda1d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:) :: ia
        call fpup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda2d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:) :: ia
        call fpup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda3d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:) :: ia
        call fpup_doubles(p, ia, size(ia))
      end subroutine
      function pup_issz(p)
        INTEGER :: p
        logical pup_issz
        pup_issz = fpup_issizing(p)
      end function
      function pup_ispk(p)
        INTEGER :: p
        logical pup_ispk
        pup_ispk = fpup_ispacking(p)
      end function
      function pup_isupk(p)
        INTEGER :: p
        logical pup_isupk
        pup_isupk = fpup_isunpacking(p)
      end function
      function pup_isdel(p)
        INTEGER :: p
        logical pup_isdel
        pup_isdel = fpup_isdeleting(p)
      end function
      function pup_isul(p)
        INTEGER :: p
        logical pup_isul
        pup_isul = fpup_isuserlevel(p)
      end function
      end module



