!     Handle the details of migrating LUNs across ranks
!     basic idea:
!     
!  Application creates the registry using createRegistry()

!  Application registers a LUN with the API using the same parameters
!  it would use to open the file using registerLUN().

!  Application calls closeRegisteredLUNSForMigration() in
!  about_to_migrate

!  Application calls reopenRegisteredLUNs() in just_migrated

!  Thats is all there is to it, otherwise the normal AMPI rules apply

!  Implementation assumes it is being used for LUNs associated with files.
!  YMMV if you need to migrate other LUNs.
      module migratablelun
      implicit none
      
!     The structure:
      type, public :: lunrec
        integer :: unit
        character(len=:),allocatable :: file
        character(len=:),allocatable :: action
        integer :: offset
        logical :: active ! internal use set by about_to_migrate
        character(len=:), allocatable :: blank 
        character(len=:), allocatable :: delim 
        character(len=:), allocatable :: pad 
        character(len=:), allocatable :: convert
        character(len=:), allocatable :: status
        character(len=:), allocatable :: access
        character(len=:), allocatable ::form
        character(len=:), allocatable :: position
        integer:: recl
        
      end type lunrec
      
!      private LUNRegistry

!     API provides migratability for each LUN by storing necessary
!     metadata to reopen after migration.

!     Note: this is orthogonal to rank based privatization via CK_LUN
!     So this module can be used with or without that, though you
!     will need to do something for lun uniqueness if you do not use CK_LUN.

!     In about_to_migrate, each registered lun should be checked via
!     inquire to assess how it is being used (open etc.)
!     then that status is recorded so it can be reproduced after migration
!     then the lun is closed
      
!     In just_migrated, each registered lun should be updated to reopen the
!     ones which were open and bring them to the same state they were in at
!     the prior to migration close point

!     Note, fseek and ftell aren't standard in fortran, but do seem to
!     be supported in basically every fortran compiler.  Because they're
!     incredibly useful and their functionality is not provided in
!     standard fortran I/O.  Which just seems like a bizarre oversight
!     given the otherwise overstuffed approach that fortran has to I/O.

!     Note: the actual action (read, write, readwrite) used to open a
!     LUN is *not* returned by inquire and doesn't seem to be
!     accessible based on the LUN via any API, therefore we just have
!     our registration take all the parameters that OPEN does and
!     store them.

!     This makes it seem like it shouldn't be that big a deal to do a
!     wrapper for open calls.  Where we just store the argument list and
!     pass it to the fortran open.  Maybe named AMPI_Open ?  Then test
!     status at migration to close if active and store offset, and
!     reopen if it was open.  Meaning, we shouldn't need to actually
!     trap the rest of the fortran I/O calls.  The rest of the API could
!     be tucked in to the AMPI initialization process and automatically
!     added to the migration callback hooks.


      type(lunrec), allocatable :: LUNRegistry(:)
!$omp threadprivate(LUNRegistry)      
      integer lastregistry;
!$omp threadprivate(lastregistry)
      contains      

      subroutine createRegistry(size)
        implicit NONE      
        integer, intent(in) :: size

        allocate(LUNRegistry(size))
        lastregistry=0;
      end subroutine createRegistry
      
      function registerLUN (unit, file, action, status, form,&
           & access, position, blank, pad, delim, convert,&
           & recl ) result(handle)
        implicit NONE
        integer, intent(in) :: unit
        character(len=*), intent(in) :: file
        character(len=*), intent(in) :: action
        ! support for fortran's headache inducing list of optional
        ! parameters for open. For API simplicity we take them in the
        ! same form OPEN does, set them to the correct default if unset,
        ! and just pass them through. Exceptions would be position
        ! and status which need correction on reopen after migration
        character(len=*),intent(in),optional :: status
        character(len=*),intent(in),optional :: form
        character(len=*),intent(in),optional :: access
        character(len=*),intent(in),optional :: position
        character(len=*),intent(in),optional :: blank
        character(len=*),intent(in),optional :: pad
        character(len=*),intent(in),optional :: delim
        character(len=*),intent(in),optional :: convert
        integer,intent(in),optional :: recl

        logical :: active
        integer :: handle
        type(lunrec) :: r
        
        r%unit=unit;
        r%file=file;
        r%action=action;
        r%offset=0;
        lastregistry = lastregistry + 1;
        ! long ugly block to handle the idiosyncracies of defaults
        if(present(status)) then
           r%status=status;
        else
           r%status='UNKNOWN'
        end if
        if(present(access)) then
           r%access=access
        else
           r%access='SEQUENTIAL'
        end if
        if(present(form)) then
           r%form=form;
        else
           if(r%access .eq. 'SEQUENTIAL')then
              r%form='FORMATTED';
           else
              r%form='UNFORMATTED';
           end if
        end if
        if(present(position)) then
           r%position=position;
        else
           r%position='ASIS';
        end if
        if(present(blank)) then
           r%blank=blank;
        else
           r%blank='NULL';
        end if
        if(present(delim)) then
           r%delim=delim;
        else
           r%delim='NONE';
        end if
        if(present(pad)) then
           r%pad=pad;
        else
           r%pad='YES';
        end if
        if(present(convert)) then
           r%convert=convert;
        else
           r%convert='NATIVE';
        end if
        if(present(recl)) then
           r%recl=recl;
        else
           r%recl=-1;
        end if
        handle=lastregistry;
        lunregistry(handle)=r;
        return
      end function registerLUN

      function closeRegisteredLUNsForMigration() result(count)
        implicit NONE
        integer:: count
        DO count=1, lastregistry
!     if registered lun is open, get the offset, close it
!     use inquire to test for LUN open
           INQUIRE( UNIT=LUNRegistry(count)%unit, OPENED = LUNRegistry(count)%active);
           IF ( LUNRegistry(count)%active ) THEN
              LUNRegistry(count)%offset = FTELL(LUNRegistry(count)%unit);
              CLOSE ( LUNRegistry(count)%unit );
           END IF

        END DO
        return
      end function closeRegisteredLUNsForMigration

      function reopenRegisteredLUNs() result(ierr)
        implicit NONE
        integer ierr, count
        character:: iostring
        DO count=1, lastregistry

!     if the action is read, or readwrite seek to the offset after opening
!           print *,"checking registry entry ",count, " of ", lastregistry, " file ",trim(adjustl(LUNRegistry(count)%file));
           IF(LUNRegistry(count)%active) THEN
              IF(trim(adjustl(LUNRegistry(count)%action)).eq.'WRITE') THEN
                 ! modify the position & status
                 LUNRegistry(count)%position='APPEND';
                 LUNRegistry(count)%status='OLD';
              END IF
              if(LUNRegistry(count)%recl .gt. 0) THEN
              OPEN(UNIT=LUNRegistry(count)%unit, &
                   & FILE=LUNRegistry(count)%file, &
                   & IOSTAT=ierr, &
                   & STATUS=LUNRegistry(count)%status, &
                   & ACTION=LUNRegistry(count)%action, &
                   & FORM=LUNRegistry(count)%form, &
                   & DELIM=LUNRegistry(count)%delim, &
                   & PAD=LUNRegistry(count)%pad, &
                   & CONVERT=LUNRegistry(count)%convert, &
                   & RECL=LUNRegistry(count)%recl, &
                   & ACCESS=LUNRegistry(count)%ACCESS); 
              ELSE
                 OPEN(UNIT=LUNRegistry(count)%unit, &
                      & FILE=LUNRegistry(count)%file, &
                      & IOSTAT=ierr, &
                      & STATUS=LUNRegistry(count)%status, &
                      & ACTION=LUNRegistry(count)%action, &
                      & FORM=LUNRegistry(count)%form, &
                      & DELIM=LUNRegistry(count)%delim, &
                      & PAD=LUNRegistry(count)%pad, &
                      & CONVERT=LUNRegistry(count)%convert, &
                      & ACCESS=LUNRegistry(count)%ACCESS); 
              END IF
              IF( ierr .NE. 0) THEN
                 PRINT *,"error opening ",LUNRegistry(count)%file," err " &
                      ," action ", LUNRegistry(count)%action &
                      ,ierr, iostring;
                 RETURN
              END IF
              CALL FSEEK(LUNRegistry(count)%unit, LUNRegistry(count)%offset, 0, ierr);
              IF( ierr .NE. 0) THEN
                 PRINT *,"error fseek", trim(adjustl(LUNRegistry(count)%file)),"err", ierr;
     
                 RETURN
              END IF
           ELSE
              write(*,*) "not reopening inactive ",trim(adjustl(LUNRegistry(count)%file));
           END IF
        END DO
        RETURN
      end function reopenRegisteredLUNs
      
      END MODULE migratablelun
      
      
