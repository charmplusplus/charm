      PROGRAM main1

      character*80 argv(0:99)
      integer argc, length(0:99)
      integer i,ierr
      
!      argc = ipxfargc()
      
!      do i = 0,argc
!         call pxfgetarg(i,argv(i),length(i),ierr)
!         if (ierr .ne. 0) print*,'Arg ',i,' error'
!      end do

      argc = IARGC()
      do i = 0,argc
         call GETARG(i,argv(i))
         length(i) = LEN_TRIM(argv(i));
      end do

      call conversemain(argc,argv,length)

      END

