      module pupmod
      implicit none
      external pup_int
      external pup_ints
      external pup_char
      external pup_chars
      external pup_short
      external pup_shorts
      external pup_real
      external pup_reals
      external pup_double
      external pup_doubles
      interface
        function pup_isSizing(p)
          INTEGER :: p
          logical pup_isSizing
        end function
        function pup_isPacking(p)
          INTEGER :: p
          logical pup_isPacking
        end function
        function pup_isUnpacking(p)
          INTEGER :: p
          logical pup_isUnpacking
        end function
        function pup_isDeleting(p)
          INTEGER :: p
          logical pup_isDeleting
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
        call pup_int(p, i)
      end subroutine
      subroutine pia1d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia2d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:,:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia3d(p, ia)
        INTEGER :: p
        integer, intent(inout), dimension(:,:,:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pc(p, i)
        INTEGER :: p
        character*1, intent(inout) :: i
        call pup_char(p, i)
      end subroutine
      subroutine pca1d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca2d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:,:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca3d(p, ia)
        INTEGER :: p
        character*1, intent(inout), dimension(:,:,:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine ps(p, i)
        INTEGER :: p
        integer(kind=2), intent(inout) :: i
        call pup_short(p, i)
      end subroutine
      subroutine psa1d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa2d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa3d(p, ia)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine pr(p, i)
        INTEGER :: p
        real(kind=4), intent(inout) :: i
        call pup_real(p, i)
      end subroutine
      subroutine pra1d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra2d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra3d(p, ia)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pd(p, i)
        INTEGER :: p
        real(kind=8), intent(inout) :: i
        call pup_double(p, i)
      end subroutine
      subroutine pda1d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda2d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda3d(p, ia)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      function pup_issz(p)
        INTEGER :: p
        logical pup_issz
        pup_issz = pup_isSizing(p)
      end function
      function pup_ispk(p)
        INTEGER :: p
        logical pup_ispk
        pup_ispk = pup_isPacking(p)
      end function
      function pup_isupk(p)
        INTEGER :: p
        logical pup_isupk
        pup_isupk = pup_isUnpacking(p)
      end function
      function pup_isdel(p)
        INTEGER :: p
        logical pup_isdel
        pup_isdel = pup_isDeleting(p)
      end function
      end module



