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
          external p
          integer pup_isSizing
        end function
        function pup_isPacking(p)
          external p
          integer pup_isPacking
        end function
        function pup_isUnpacking(p)
          external p
          integer pup_isUnpacking
        end function
      end interface
      interface pup
        module procedure pi,pia1d,pia2d,pia3d,pc,pca1d,pca2d,pca3d
        module procedure ps,psa1d,psa2d,psa3d,pr,pra1d,pra2d,pra3d
        module procedure pd,pda1d,pda2d,pda3d
      end interface
contains
      subroutine pi(p, i)
        external p
        integer, intent(inout) :: i
        call pup_int(p, i)
      end subroutine
      subroutine pia1d(p, ia)
        external p
        integer, intent(inout), dimension(:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia2d(p, ia)
        external p
        integer, intent(inout), dimension(:,:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pia3d(p, ia)
        external p
        integer, intent(inout), dimension(:,:,:) :: ia
        call pup_ints(p, ia, size(ia))
      end subroutine
      subroutine pc(p, i)
        external p
        character*1, intent(inout) :: i
        call pup_char(p, i)
      end subroutine
      subroutine pca1d(p, ia)
        external p
        character*1, intent(inout), dimension(:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca2d(p, ia)
        external p
        character*1, intent(inout), dimension(:,:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine pca3d(p, ia)
        external p
        character*1, intent(inout), dimension(:,:,:) :: ia
        call pup_chars(p, ia, size(ia))
      end subroutine
      subroutine ps(p, i)
        external p
        integer(kind=2), intent(inout) :: i
        call pup_short(p, i)
      end subroutine
      subroutine psa1d(p, ia)
        external p
        integer(kind=2), intent(inout), dimension(:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa2d(p, ia)
        external p
        integer(kind=2), intent(inout), dimension(:,:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine psa3d(p, ia)
        external p
        integer(kind=2), intent(inout), dimension(:,:,:) :: ia
        call pup_shorts(p, ia, size(ia))
      end subroutine
      subroutine pr(p, i)
        external p
        real(kind=4), intent(inout) :: i
        call pup_real(p, i)
      end subroutine
      subroutine pra1d(p, ia)
        external p
        real(kind=4), intent(inout), dimension(:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra2d(p, ia)
        external p
        real(kind=4), intent(inout), dimension(:,:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pra3d(p, ia)
        external p
        real(kind=4), intent(inout), dimension(:,:,:) :: ia
        call pup_reals(p, ia, size(ia))
      end subroutine
      subroutine pd(p, i)
        external p
        real(kind=8), intent(inout) :: i
        call pup_double(p, i)
      end subroutine
      subroutine pda1d(p, ia)
        external p
        real(kind=8), intent(inout), dimension(:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda2d(p, ia)
        external p
        real(kind=8), intent(inout), dimension(:,:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      subroutine pda3d(p, ia)
        external p
        real(kind=8), intent(inout), dimension(:,:,:) :: ia
        call pup_doubles(p, ia, size(ia))
      end subroutine
      function pup_issz(p)
        external p
        logical pup_issz
        pup_issz = (pup_isSizing(p) .eq. 1)
      end function
      function pup_ispk(p)
        external p
        logical pup_ispk
        pup_ispk = (pup_isPacking(p) .eq. 1)
      end function
      function pup_isupk(p)
        external p
        logical pup_isupk
        pup_isupk = (pup_isUnpacking(p) .eq. 1)
      end function
end module
