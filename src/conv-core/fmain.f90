PROGRAM main1

  character*80 argv(0:99)
  integer argc, length(0:99)
  integer i,ierr

  argc = ipxfargc()

  do i = 0,argc
     call pxfgetarg(i,argv(i),length(i),ierr)
     if (ierr .ne. 0) print*,'Arg ',i,' error'
  end do
  call fmain(argc,argv,length)

END
