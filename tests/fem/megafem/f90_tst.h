#define C(x)
C ( Bizarre header to create ".tst" source code that )
C ( can be compiled as either Fortran *or* C. )
C ( Include this header to preprocess the code into F90. )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/22 )
#define TST_F90 1  C( We are building for fortran)
#define IDXBASE 1  C(Fortran arrays start at 1)

#define STRING CHARACTER(LEN=*) 

#define CREATE_TYPE(typeName) TYPE typeName 
#define TYPE_POINTER(typeName) TYPE(typeName)
C ( TYPE(name) is unmodified )
#define END_TYPE END TYPE

#define DECLARE_ARRAY(type,var) type, POINTER :: var(:)
#define ALLOCATE_ARRAY(var,size) ALLOCATE(var(size))

#define DECLARE_ARRAY2D(type,var,nearSize) type, POINTER :: var(:,:)
#define ALLOCATE_ARRAY2D(var,nearSize,size) ALLOCATE(var(nearSize,size))

C ( Declare subroutines with various numbers of arguments )
#define SUBROUTINE0(name) \
SUBROUTINE name;\
   TST_F90_USE; IMPLICIT NONE;

#define SUBROUTINE1(name, t1,a1) \
SUBROUTINE name(a1);\
   TST_F90_USE;\
   IMPLICIT NONE;\
   t1 a1;

#define SUBROUTINE2(name, t1,a1, t2,a2) \
SUBROUTINE name(a1,a2); \
   TST_F90_USE; IMPLICIT NONE; \
   t1 a1; t2 a2;

#define SUBROUTINE3(name, t1,a1, t2,a2, t3,a3) \
SUBROUTINE name(a1,a2,a3); \
   TST_F90_USE; IMPLICIT NONE; \
   t1 a1; t2 a2; t3 a3;

#define SUBROUTINE4(name, t1,a1, t2,a2, t3,a3, t4,a4) \
SUBROUTINE name(a1,a2,a3,a4); \
   TST_F90_USE; IMPLICIT NONE; \
   t1 a1; t2 a2; t3 a3; t4 a4;

#define SUBROUTINE5(name, t1,a1, t2,a2, t3,a3, t4,a4, t5,a5) \
SUBROUTINE name(a1,a2,a3,a4,a5); \
   TST_F90_USE; IMPLICIT NONE; \
   t1 a1; t2 a2; t3 a3; t4 a4; t5 a5;


C ( Loop the value of variable from 1..upper )
#define FOR(variable,upper) \
    DO variable=1,upper
#define END_FOR END DO

C ( "IF (condition) THEN" is unmodified )
#define END_IF END IF

#define _AND_ .AND.
#define _OR_ .OR.
#define _NE_ .NE.
#define ADDR 
#define v %
