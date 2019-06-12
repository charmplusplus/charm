### AMPI-zation tutorial:
To convert a given MPI program and make it AMPI compatible, the program needs to be made thread-safe.
> Note: This tutorial only covers the TLSglobals method. The other approaches to make a program thread-safe can be found in the global_variable_privatisation.pdf (The link will be posted shortly).

This tutorial details one of the conversion processes through the use of an MPI program called NAS Parallel Benchmarks found here: https://www.nas.nasa.gov/publications/npb.html
To recreate this example, clone this repository, and checkout the commit version you are interested in (specifically the commit where the MPI version of a specific code sample was uploaded.) Next, find the location of the relevant charm build and update the CHARMDIR variable in the following files:
- ./common.mk
- ./NPB3.3/config/make.def
After this, go ahead and compile the NPB program. For our purposes, the following build command should suffice:
```
make is NPROCS=32 CLASS=W
```
The executable named is-W-32 should be produced in the ./NPB-3.3/bin/ folder. Next, we make use of the nm_gloabls.sh symlink provided inside this folder. This script uses the nm utility to retrieve a list of writable global variables that could potentially make your MPI program 'thread-unsafe'.

The following code snippet is the output produced when the script nm_globals.sh is run on NAS Parallel Benchmarks v3.3 MPI:
```
The following global writable variables in './NPB3.3/bin/is.W.32' were found:

File                       Type          Name
./NPB3.3-MPI/bin/is.W.32:  D              A_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              A_test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  D              B_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              B_test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  B              bucket_ptrs   
./NPB3.3-MPI/bin/is.W.32:  B              bucket_size   
./NPB3.3-MPI/bin/is.W.32:  B              bucket_size_totals   
./NPB3.3-MPI/bin/is.W.32:  B              comm_size   
./NPB3.3-MPI/bin/is.W.32:  D              C_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              C_test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  D              D_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              D_test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  B              elapsed   
./NPB3.3-MPI/bin/is.W.32:  B              key_array   
./NPB3.3-MPI/bin/is.W.32:  B              key_buff1   
./NPB3.3-MPI/bin/is.W.32:  B              key_buff2   
./NPB3.3-MPI/bin/is.W.32:  B              key_buff_ptr_global   
./NPB3.3-MPI/bin/is.W.32:  B              my_rank   
./NPB3.3-MPI/bin/is.W.32:  B              passed_verification   
./NPB3.3-MPI/bin/is.W.32:  B              process_bucket_distrib_ptr1   
./NPB3.3-MPI/bin/is.W.32:  B              process_bucket_distrib_ptr2   
./NPB3.3-MPI/bin/is.W.32:  B              recv_count   
./NPB3.3-MPI/bin/is.W.32:  B              recv_displ   
./NPB3.3-MPI/bin/is.W.32:  B              send_count   
./NPB3.3-MPI/bin/is.W.32:  B              send_displ   
./NPB3.3-MPI/bin/is.W.32:  B              start   
./NPB3.3-MPI/bin/is.W.32:  D              S_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              S_test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  B              test_index_array   
./NPB3.3-MPI/bin/is.W.32:  B              test_rank_array   
./NPB3.3-MPI/bin/is.W.32:  B              timeron   
./NPB3.3-MPI/bin/is.W.32:  B              total_lesser_keys   
./NPB3.3-MPI/bin/is.W.32:  B              total_local_keys   
./NPB3.3-MPI/bin/is.W.32:  D              W_test_index_array   
./NPB3.3-MPI/bin/is.W.32:  D              W_test_rank_array   

Legend:
  B - The symbol is in the uninitialized data section (BSS).
  D - The symbol is in the initialized data section.
  G - The symbol is in the initialized data section for small objects.
  S - The symbol is in the uninitialized data section for small objects.
```
In order to privatise these variables using AMPI's tlsglobals support, we simply add the following call:
```
#pragma omp threadprivate()
```
This is all that is needed to privatise this program.

The following code snippet is taken from ./NPB3.3/IS/is.c lines 260-284 (AMPI version). This small addition to the code base allows users to run this MPI program on AMPI.
```c
  /****************************************************************/
	/* Global privatization using openMP's threadprivate directive  */
	/****************************************************************/
	#pragma omp threadprivate (timeron)
	#pragma omp threadprivate (my_rank)
	#pragma omp threadprivate (comm_size)
	#pragma omp threadprivate (passed_verification)
	#pragma omp threadprivate (key_buff_ptr_global)
	#pragma omp threadprivate (total_local_keys)
	#pragma omp threadprivate (key_array)
	#pragma omp threadprivate (key_buff1)
	#pragma omp threadprivate (key_buff2)
	#pragma omp threadprivate (bucket_size)
	#pragma omp threadprivate (bucket_size_totals)
	#pragma omp threadprivate (bucket_ptrs)
	#pragma omp threadprivate (process_bucket_distrib_ptr1)
	#pragma omp threadprivate (process_bucket_distrib_ptr2)
	#pragma omp threadprivate (send_count)
	#pragma omp threadprivate (recv_count)
	#pragma omp threadprivate (send_displ)
	#pragma omp threadprivate (recv_displ)
	#pragma omp threadprivate (test_index_array)
	#pragma omp threadprivate (test_rank_array)
```
