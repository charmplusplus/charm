
Entry points are assigned disjointly (i.e. all the host and node EPs
are distinct.)

0..E1 : 	System Host EPs
E1+1 .. E2: 	User Host Eps
E2+1 .. E3:	System Node Eps
E3+1 .. E4: 	User Node Eps

The system Eps are fixed before compilation (of user programs),
and so will be simply #defined in chare.h
#define NumSysEpsNode 0	 /* i.e. E1+1 */
#define NumSysEpsHost 0  /* i.e. E3-E2 */

(conversely: E1 = NumSysEpsNode-1,

The number of user entry points can also be #defined (by the translator)
in the file ???.? (MANISH, which one?)
But as ckobj.o must be pre-compiled independent of the user program,
#define won't work.
So: global variables will be used:

NumUserEpsNode and NumUserEpsHost will be set by the user
in the initialization functions in main programs on the host and node.


These will be used to allocate tables, but also to decide
the decrement in EP to be made before accessing the table-of-function-pointers
for a given EP on the node. 
(Note that the table on a node goes from 0 .. NumSysEpsNode + NumUserEpsNode -1
but the EPs range from NumSysEpsHost + NumUserEpsHost upwards.
So this sum must be substracted from the EP for accessing the table.)
