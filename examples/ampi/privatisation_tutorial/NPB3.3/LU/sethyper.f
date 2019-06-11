
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine sethyper

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c    for each column in a hyperplane, istart = first row,
c---------------------------------------------------------------------

      implicit none

      include 'applu.incl'

c---------------------------------------------------------------------
c  local variables
c---------------------------------------------------------------------
      integer i, j
      integer iglob, jglob
      integer kp

c---------------------------------------------------------------------
c compute the pointers for hyperplanes
c---------------------------------------------------------------------
        do kp = 2,nx0+ny0
          icomms(kp) = .false.
          icommn(kp) = .false.
          icomme(kp) = .false.
          icommw(kp) = .false.

c---------------------------------------------------------------------
c  check to see if comm. to south is required
c---------------------------------------------------------------------
          if (south.ne.-1) then
            i     = iend
            iglob = ipt + i
            jglob = kp - iglob
            j     = jglob - jpt
            if (jglob.ge.2.and.jglob.le.ny0-1.and.j.ge.jst.and.
     >         j.le.jend) icomms(kp) = .true.
          end if

c---------------------------------------------------------------------
c  check to see if comm. to north is required
c---------------------------------------------------------------------
          if (north.ne.-1) then
            i     = ist
            iglob = ipt + i
            jglob = kp - iglob
            j     = jglob - jpt
            if (jglob.ge.2.and.jglob.le.ny0-1.and.j.ge.jst.and.
     >         j.le.jend) icommn(kp) = .true.
          end if

c---------------------------------------------------------------------
c  check to see if comm. to east is required
c---------------------------------------------------------------------
          if (east.ne.-1) then
            j     = jend
            jglob = jpt + j
            iglob = kp - jglob
            i     = iglob - ipt
            if (iglob.ge.2.and.iglob.le.nx0-1.and.i.ge.ist.and.
     >         i.le.iend) icomme(kp) = .true.
          end if

c---------------------------------------------------------------------
c  check to see if comm. to west is required
c---------------------------------------------------------------------
          if (west.ne.-1) then
            j = jst
            jglob = jpt + j
            iglob = kp - jglob
            i     = iglob - ipt
            if (iglob.ge.2.and.iglob.le.nx0-1.and.i.ge.ist.and.
     >         i.le.iend) icommw(kp) = .true.
          end if

        end do

        icomms(1) = .false.
        icommn(1) = .false.
        icomme(1) = .false.
        icommw(1) = .false.
        icomms(nx0+ny0+1) = .false.
        icommn(nx0+ny0+1) = .false.
        icomme(nx0+ny0+1) = .false.
        icommw(nx0+ny0+1) = .false.

      return
      end
