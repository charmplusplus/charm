!     DON NOT EDIT THIS FILE, GENERATE IT FROM RUNNING pup_f.f90.sh
      module pupmod
      implicit none
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

        subroutine fpup_char(p, d)
          INTEGER :: p
          CHARACTER :: d
        end subroutine
        subroutine fpup_short(p, d)
          INTEGER :: p
          INTEGER (KIND=2) :: d
        end subroutine
        subroutine fpup_int(p, d)
          INTEGER :: p
          INTEGER (KIND=4) :: d
        end subroutine
        subroutine fpup_long(p, d)
          INTEGER :: p
          INTEGER (KIND=8) :: d
        end subroutine
        subroutine fpup_real(p, d)
          INTEGER :: p
          REAL (KIND=4)  :: d
        end subroutine
        subroutine fpup_double(p, d)
          INTEGER :: p
          REAL (KIND=8)  :: d
        end subroutine
        subroutine fpup_logical(p, d)
          INTEGER :: p
          LOGICAL :: d
        end subroutine
      end interface

      interface fpup_chars
       module procedure fpup_chars_0
       module procedure fpup_chars_1
       module procedure fpup_chars_2
       module procedure fpup_chars_3
       module procedure fpup_chars_4
       module procedure fpup_chars_5
       module procedure fpup_chars_6
       module procedure fpup_chars_7
      end interface fpup_chars

      interface fpup_ints
       module procedure fpup_ints_1
       module procedure fpup_ints_2
       module procedure fpup_ints_3
       module procedure fpup_ints_4
       module procedure fpup_ints_5
       module procedure fpup_ints_6
       module procedure fpup_ints_7
      end interface fpup_ints

      interface fpup_longs
       module procedure fpup_longs_1
       module procedure fpup_longs_2
       module procedure fpup_longs_3
       module procedure fpup_longs_4
       module procedure fpup_longs_5
       module procedure fpup_longs_6
       module procedure fpup_longs_7
      end interface fpup_longs

      interface fpup_reals
       module procedure fpup_reals_1
       module procedure fpup_reals_2
       module procedure fpup_reals_3
       module procedure fpup_reals_4
       module procedure fpup_reals_5
       module procedure fpup_reals_6
       module procedure fpup_reals_7
      end interface fpup_reals

      interface fpup_doubles
       module procedure fpup_doubles_1
       module procedure fpup_doubles_2
       module procedure fpup_doubles_3
       module procedure fpup_doubles_4
       module procedure fpup_doubles_5
       module procedure fpup_doubles_6
       module procedure fpup_doubles_7
      end interface fpup_doubles

      interface fpup_logicals
       module procedure fpup_logicals_1
       module procedure fpup_logicals_2
       module procedure fpup_logicals_3
       module procedure fpup_logicals_4
       module procedure fpup_logicals_5
       module procedure fpup_logicals_6
       module procedure fpup_logicals_7
      end interface fpup_logicals

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

      subroutine fpup_complex(p,c)
        INTEGER p
        complex c
        call fpup_real(p,REAL(c))
        call fpup_real(p,AIMAG(c))
      end subroutine

      subroutine fpup_complexes(p,c,size)
        INTEGER p
        complex,pointer,dimension(:) :: c
        integer size
        integer i
        do i = 1, size, 1
          call fpup_complex(p,c(i))
        end do
      end subroutine

      subroutine fpup_doublecomplex(p,c)
        INTEGER p
        double complex c
        call fpup_double(p,DBLE(c))
        call fpup_double(p,DIMAG(c))
      end subroutine

      subroutine fpup_doublecomplexes(p,c,size)
        INTEGER p
        double complex,pointer,dimension(:) :: c
        integer size
        integer i
        do i = 1, size, 1
          call fpup_doublecomplex(p,c(i))
        end do
      end subroutine

      subroutine fpup_chars_0(p, d, c)
        INTEGER :: p
        CHARACTER(LEN=*)     d
        INTEGER :: c
        call fpup_charsg(p, d, c)
      end subroutine
       subroutine fpup_chars_1(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_2(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_3(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_4(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_5(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_6(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine
       subroutine fpup_chars_7(p, d, c)
        INTEGER :: p
        character, intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_charsg(p, d, c)
       end subroutine

       subroutine fpup_shorts_1(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_2(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_3(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_4(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_5(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_6(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine
       subroutine fpup_shorts_7(p, d, c)
        INTEGER :: p
        integer(kind=2), intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_shortsg(p, d, c)
       end subroutine

       subroutine fpup_ints_1(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_2(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_3(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_4(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_5(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_6(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine
       subroutine fpup_ints_7(p, d, c)
        INTEGER :: p
        integer(kind=4), intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_intsg(p, d, c)
       end subroutine

       subroutine fpup_longs_1(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_2(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_3(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_4(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_5(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_6(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine
       subroutine fpup_longs_7(p, d, c)
        INTEGER :: p
        integer(kind=8), intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_longsg(p, d, c)
       end subroutine

       subroutine fpup_reals_1(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_2(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_3(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_4(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_5(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_6(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine
       subroutine fpup_reals_7(p, d, c)
        INTEGER :: p
        real(kind=4), intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_realsg(p, d, c)
       end subroutine

       subroutine fpup_doubles_1(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_2(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_3(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_4(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_5(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_6(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine
       subroutine fpup_doubles_7(p, d, c)
        INTEGER :: p
        real(kind=8), intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_doublesg(p, d, c)
       end subroutine

       subroutine fpup_logicals_1(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_2(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_3(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_4(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:,:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_5(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_6(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine
       subroutine fpup_logicals_7(p, d, c)
        INTEGER :: p
        logical, intent(inout), dimension(:,:,:,:,:,:,:) :: d
        INTEGER :: c
        call fpup_logicalsg(p, d, c)
       end subroutine



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
