/* Bizarre header to create ".tst" source code that
   can be compiled as either Fortran *or* C++.
   Include this header to preprocess the code into C++.
   
   Orion Sky Lawlor, olawlor@acm.org, 2003/7/22
*/
#define TST_C 1 /* This is a C test */
#define IDXBASE 0 /* C arrays start at zero */

#define C(x) /* Comment field: empty, preprocessed away */
#define CALL /* empty, C doesn't have a CALL keyword. */

#define INTEGER int
#define SINGLE float
#define DOUBLE double
#define PRECISION /* empty, use like DOUBLE PRECISION x */
#define STRING const char *

#define CREATE_TYPE(typeName) class typeName { public:
#define END_TYPE };
#define TYPE_POINTER(typeName) typeName &
#define TYPE(typeName) typeName

/* A zero-based array that can be indexed using Fortran-
   style round braces, like arr(i). */
template <class T>
class F90styleArray {
	CkVec<T> sto;
public:
	void resize(int sz) {sto.resize(sz);}
	T &operator()(int i) {
		if (i<0 || i>=sto.size()) CkAbort("F90style array out-of-bounds!");
		return sto[i];
	}
	operator T *() {return &sto[0];}
};
#define DECLARE_ARRAY(type,var) F90styleArray<type> var
#define ALLOCATE_ARRAY(var,size) var.resize(size)
#define DEALLOCATE(var) /* empty, destructor does work */

/* A zero-based array that can be indexed using Fortran-
   style 2D round braces, like arr(i,j). Stored in row-major
   order, with arr(i,j) and arr(i+1,j) contiguous. */
template <class T,int n>
class F90styleArray2D {
	F90styleArray<T> sto;
public:
	void resize(int sz) { sto.resize(n*sz);}
	T &operator()(int i,int j) {
		if (i<0 || i>=n) CkAbort("F90style2d array out-of-bounds!\n");
		return sto[i+j*n];
	}
	operator T *() {return sto;}
};
#define DECLARE_ARRAY2D(type,var,nearSize) F90styleArray2D<type,nearSize> var
#define ALLOCATE_ARRAY2D(var,nearSize,size) var.resize(size)


/* Declare subroutines with various numbers of arguments */
#define SUBROUTINE0(name) \
void name() {

#define SUBROUTINE1(name, t1,a1) \
void name(t1 a1) {

#define SUBROUTINE2(name, t1,a1, t2,a2) \
void name(t1 a1, t2 a2) {

#define SUBROUTINE3(name, t1,a1, t2,a2, t3,a3) \
void name(t1 a1, t2 a2, t3 a3) {

#define SUBROUTINE4(name, t1,a1, t2,a2, t3,a3, t4,a4) \
void name(t1 a1, t2 a2, t3 a3, t4 a4) {

#define SUBROUTINE5(name, t1,a1, t2,a2, t3,a3, t4,a4, t5,a5) \
void name(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5) {

#define END } /* close subroutine */

/* Loop the value of variable from 0..upper-1 */
#define FOR(variable,upper) \
	for (variable=0;variable<upper;variable++) {
#define END_FOR }

#define IF if
#define THEN {
#define ELSE } else {
#define END_IF }

#define _AND_ &&
#define _OR_ ||
#define _NE_ !=
#define ADDR & /* Address-of operation */
#define v . /* Extract-member-of-structure operation */

