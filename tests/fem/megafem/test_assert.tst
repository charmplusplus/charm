C ( C/Fortran test driver: Overall driver )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/22 )

C ( C/Fortran test routines: assertations and test utilities )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/25 )

C ( ------------------ Utility routines ------------------- )
SUBROUTINE1(TST_assert_isnt_negative, INTEGER,val)
  IF (val < 0) THEN
    CALL RUN_Abort(val);
  END_IF
END
SUBROUTINE1(TST_assert_is_index, INTEGER,val)
  IF (val < IDXBASE) THEN
    CALL RUN_Abort(val);
  END_IF
END
SUBROUTINE3(TST_assert_range, INTEGER,val, INTEGER,l, INTEGER,h)
  IF (val<l _OR_ val>=h) THEN
    CALL RUN_Abort(val);
  END_IF
END
SUBROUTINE2(TST_assert_index_range, INTEGER,val, INTEGER,m)
  CALL TST_assert_range(val, IDXBASE,m+IDXBASE);
END
SUBROUTINE2(TST_assert_is_chunk, TYPE_POINTER(global),g, INTEGER,val)
  CALL TST_assert_range(val,0,g v commSize);
END
SUBROUTINE2(TST_assert_equal, INTEGER,val, INTEGER,should)
  IF (val _NE_ should) THEN
    CALL RUN_Abort(val);
  END_IF
END

#ifdef TST_C
void TST_print_str(const char *desc) {
	CkPrintf("[%d] %s\n",FEM_My_partition(),desc); 
}
void TST_print_str2(const char *desc,const char *desc2) {
	CkPrintf("[%d] %s %s\n",FEM_My_partition(),desc,desc2); 
}
void TST_print_int(const char *desc,int i) { 
	CkPrintf("[%d] %s: %d\n",FEM_My_partition(),desc,i); 
}
void TST_print_int2(const char *desc,int i,int j) { 
	CkPrintf("[%d] %s: %d %d\n",FEM_My_partition(),desc,i,j); 
}
void TST_print_int3(const char *desc,int i,int j,int k) { 
	CkPrintf("[%d] %s: %d %d %d\n",FEM_My_partition(),desc,i,j,k); 
}
#else /* Fortran */
SUBROUTINE TST_print_str(desc)
  TST_F90_USE; STRING :: desc;
  write(*,*) '[',FEM_My_partition()-1,'] ',desc;
END SUBROUTINE
SUBROUTINE TST_print_str2(desc,desc2)
  TST_F90_USE
  CHARACTER(LEN=*) :: desc, desc2;
  write(*,*) '[',FEM_My_partition()-1,'] ',desc,' ',desc2;
END SUBROUTINE
SUBROUTINE TST_print_int(desc,i)
  TST_F90_USE
  CHARACTER(LEN=*) :: desc; INTEGER i;
  write(*,*) '[',FEM_My_partition()-1,'] ',desc,':',i;
END SUBROUTINE
SUBROUTINE TST_print_int2(desc,i,j)
  TST_F90_USE
  CHARACTER(LEN=*) :: desc; INTEGER i,j;
  write(*,*) '[',FEM_My_partition()-1,'] ',desc,':',i,j;
END SUBROUTINE
SUBROUTINE TST_print_int3(desc,i,j,k)
  TST_F90_USE
  CHARACTER(LEN=*) :: desc; INTEGER i,j,k;
  write(*,*) '[',FEM_My_partition()-1,'] ',desc,':',i,j,k;
END SUBROUTINE
#endif

SUBROUTINE1(TST_status, STRING,routine)
  CALL TST_print_str2("testing",routine);
END
