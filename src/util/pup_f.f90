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
      interface apup
        module procedure apia1d,apia2d,apia3d, apca1d,apca2d,apca3d
        module procedure apsa1d,apsa2d,apsa3d, apra1d,apra2d,apra3d
        module procedure apda1d,apda2d,apda3d
      end interface
      contains
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



      subroutine pi(p, i)
        INTEGER :: p
        integer, intent(inout) :: i
        call fpup_int(p, i)
      end subroutine
      
      subroutine pia1d(p, arr)
        INTEGER :: p
        integer, intent(inout), dimension(:) :: arr
        call fpup_ints(p, arr, size(arr))
      end subroutine
      subroutine pia2d(p, arr)
        INTEGER :: p
        integer, intent(inout), dimension(:,:) :: arr
        call fpup_ints(p, arr, size(arr))
      end subroutine
      subroutine pia3d(p, arr)
        INTEGER :: p
        integer, intent(inout), dimension(:,:,:) :: arr
        call fpup_ints(p, arr, size(arr))
      end subroutine
      
      subroutine apia1d(p, arr)
        INTEGER :: p
        integer, pointer, dimension(:) :: arr
        integer :: n(1)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,1)
          ALLOCATE(arr(n(1)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          CALL fpup_ints(p,n,1);
        END IF
        call fpup_ints(p, arr, n(1))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apia2d(p, arr)
        INTEGER :: p
        integer, pointer, dimension(:,:) :: arr
        integer :: n(2)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,2)
          ALLOCATE(arr(n(1),n(2)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          CALL fpup_ints(p,n,2);
        END IF
        call fpup_ints(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apia3d(p, arr)
        INTEGER :: p
        integer, pointer, dimension(:,:,:) :: arr
        integer :: n(3)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,3)
          ALLOCATE(arr(n(1),n(2),n(3)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          n(3)=SIZE(arr,DIM=3)
          CALL fpup_ints(p,n,3);
        END IF
        call fpup_ints(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine




      subroutine ps(p, i)
        INTEGER :: p
        integer(kind=2), intent(inout) :: i
        call fpup_short(p, i)
      end subroutine
      
      subroutine psa1d(p, arr)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:) :: arr
        call fpup_shorts(p, arr, size(arr))
      end subroutine
      subroutine psa2d(p, arr)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:) :: arr
        call fpup_shorts(p, arr, size(arr))
      end subroutine
      subroutine psa3d(p, arr)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:) :: arr
        call fpup_shorts(p, arr, size(arr))
      end subroutine
      
      subroutine apsa1d(p, arr)
        INTEGER :: p
        integer(kind=2), pointer, dimension(:) :: arr
        integer :: n(1)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,1)
          ALLOCATE(arr(n(1)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          CALL fpup_ints(p,n,1);
        END IF
        call fpup_shorts(p, arr, n(1))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apsa2d(p, arr)
        INTEGER :: p
        integer(kind=2), pointer, dimension(:,:) :: arr
        integer :: n(2)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,2)
          ALLOCATE(arr(n(1),n(2)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          CALL fpup_ints(p,n,2);
        END IF
        call fpup_shorts(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apsa3d(p, arr)
        INTEGER :: p
        integer(kind=2), pointer, dimension(:,:,:) :: arr
        integer :: n(3)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,3)
          ALLOCATE(arr(n(1),n(2),n(3)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          n(3)=SIZE(arr,DIM=3)
          CALL fpup_ints(p,n,3);
        END IF
        call fpup_shorts(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine




      subroutine pc(p, i)
        INTEGER :: p
        character, intent(inout) :: i
        call fpup_char(p, i)
      end subroutine
      
      subroutine pca1d(p, arr)
        INTEGER :: p
        character, intent(inout), dimension(:) :: arr
        call fpup_chars(p, arr, size(arr))
      end subroutine
      subroutine pca2d(p, arr)
        INTEGER :: p
        character, intent(inout), dimension(:,:) :: arr
        call fpup_chars(p, arr, size(arr))
      end subroutine
      subroutine pca3d(p, arr)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:) :: arr
        call fpup_chars(p, arr, size(arr))
      end subroutine
      
      subroutine apca1d(p, arr)
        INTEGER :: p
        character, pointer, dimension(:) :: arr
        integer :: n(1)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,1)
          ALLOCATE(arr(n(1)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          CALL fpup_ints(p,n,1);
        END IF
        call fpup_chars(p, arr, n(1))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apca2d(p, arr)
        INTEGER :: p
        character, pointer, dimension(:,:) :: arr
        integer :: n(2)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,2)
          ALLOCATE(arr(n(1),n(2)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          CALL fpup_ints(p,n,2);
        END IF
        call fpup_chars(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apca3d(p, arr)
        INTEGER :: p
        character, pointer, dimension(:,:,:) :: arr
        integer :: n(3)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,3)
          ALLOCATE(arr(n(1),n(2),n(3)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          n(3)=SIZE(arr,DIM=3)
          CALL fpup_ints(p,n,3);
        END IF
        call fpup_chars(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine




      subroutine pr(p, i)
        INTEGER :: p
        real(kind=4), intent(inout) :: i
        call fpup_real(p, i)
      end subroutine
      
      subroutine pra1d(p, arr)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:) :: arr
        call fpup_reals(p, arr, size(arr))
      end subroutine
      subroutine pra2d(p, arr)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:) :: arr
        call fpup_reals(p, arr, size(arr))
      end subroutine
      subroutine pra3d(p, arr)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:) :: arr
        call fpup_reals(p, arr, size(arr))
      end subroutine
      
      subroutine apra1d(p, arr)
        INTEGER :: p
        real(kind=4), pointer, dimension(:) :: arr
        integer :: n(1)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,1)
          ALLOCATE(arr(n(1)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          CALL fpup_ints(p,n,1);
        END IF
        call fpup_reals(p, arr, n(1))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apra2d(p, arr)
        INTEGER :: p
        real(kind=4), pointer, dimension(:,:) :: arr
        integer :: n(2)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,2)
          ALLOCATE(arr(n(1),n(2)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          CALL fpup_ints(p,n,2);
        END IF
        call fpup_reals(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apra3d(p, arr)
        INTEGER :: p
        real(kind=4), pointer, dimension(:,:,:) :: arr
        integer :: n(3)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,3)
          ALLOCATE(arr(n(1),n(2),n(3)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          n(3)=SIZE(arr,DIM=3)
          CALL fpup_ints(p,n,3);
        END IF
        call fpup_reals(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine




      subroutine pd(p, i)
        INTEGER :: p
        real(kind=8), intent(inout) :: i
        call fpup_double(p, i)
      end subroutine
      
      subroutine pda1d(p, arr)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:) :: arr
        call fpup_doubles(p, arr, size(arr))
      end subroutine
      subroutine pda2d(p, arr)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:) :: arr
        call fpup_doubles(p, arr, size(arr))
      end subroutine
      subroutine pda3d(p, arr)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:) :: arr
        call fpup_doubles(p, arr, size(arr))
      end subroutine
      
      subroutine apda1d(p, arr)
        INTEGER :: p
        real(kind=8), pointer, dimension(:) :: arr
        integer :: n(1)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,1)
          ALLOCATE(arr(n(1)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          CALL fpup_ints(p,n,1);
        END IF
        call fpup_doubles(p, arr, n(1))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apda2d(p, arr)
        INTEGER :: p
        real(kind=8), pointer, dimension(:,:) :: arr
        integer :: n(2)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,2)
          ALLOCATE(arr(n(1),n(2)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          CALL fpup_ints(p,n,2);
        END IF
        call fpup_doubles(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine
      
      subroutine apda3d(p, arr)
        INTEGER :: p
        real(kind=8), pointer, dimension(:,:,:) :: arr
        integer :: n(3)
        IF (fpup_isunpacking(p)) THEN
          CALL fpup_ints(p,n,3)
          ALLOCATE(arr(n(1),n(2),n(3)))
        ELSE
          n(1)=SIZE(arr,DIM=1)
          n(2)=SIZE(arr,DIM=2)
          n(3)=SIZE(arr,DIM=3)
          CALL fpup_ints(p,n,3);
        END IF
        call fpup_doubles(p, arr, size(arr))
        IF (fpup_isdeleting(p)) THEN
          deallocate(arr)
        END IF
      end subroutine


      end module
