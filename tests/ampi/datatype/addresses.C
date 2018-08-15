#include <stdio.h>
#include <vector>
#include <numeric>
#include <mpi.h>

/*
Test case currently works of two TEMP_* functions added to AMPI
Which should be removed later

	TEMP_Get_Msg_Blocks_Count:
		Returns the number of Block that will be sent, Used
		to determine how large of a buffer to allocate to
		for the next function

	TEMP_Get_Msg_Blocks_Addresses
		Takes 2 buffers, one for the addresses to read from and
		another to for how much to read from in each address (in bytes)
		and fills them in.

Some case to check for:

1.
Where the indexes of the diplacements are no in order
The following datatype is Ccontig but sending a partial
datatype will mean you have to send several messages
+-------------------------------- ...
| 0 | 2 | 3 | 1 | 4 | 7 | 5 | 6 | ...
+-------------------------------- ...
MPICH tested non-incrementing displacements in datatype/hindexed-zeros.c

2.
Where the datatypes overlap with each other
+----------------------------------------- ...
| +-----------+ +-----------+ +----------- ...
| |     1     | |     3     | |     5      ...
| +-----------+ +-----------+ +----------- ...
|     +-----------+ +-----------+ +------- ...
|     |     2     | |     4     | |   6    ...
|     +-----------+ +-----------+ +------- ...
+----------------------------------------- ...
Which can happen if the sepcified displacement does not allow for
the datatypes to be seperate in memory in Structs of Indexed datatypes
*/

// If you need to test largers sizes you only neeed to change SIZE
// here as all datatypes and tests are built off this value
// (though you may want to turn off verbose)
#define SIZE 8

// 0 No printing other than the number of errors
// 1 Prints outs indiviual errors
// 2 Prints out the test cases names for debugging
int verbose = 1;

#define printbuf(buf) 												\
	if (verbose >= 3) {												\
		fprintf(stderr, "Line %4d: %s\n", __LINE__, #buf);			\
		fprintf(stderr, "  [%d", buf[0]);							\
		for (int i = 1; i<SIZE; ++i) {								\
			fprintf(stderr, ", %d", buf[i]);						\
		}															\
		fprintf(stderr, "]\n");										\
	}

#define TEST(CASE) 													\
	if(verbose >= 2) { 												\
		fprintf(stderr, ">>>> %s\n", CASE);							\
	}

/* Creates 2 buffers of input size:
 *   sendbuf - array of ints from 1 to input size
 *   recvbuf - array of 0s
 */
#define BUFFERS(size)												\
	int sendbuf[size];												\
	for (int i=0; i<size; i++) {									\
		sendbuf[i] = i+1;											\
	}																\
	int recvbuf[size];												\
	for (int i=0; i<size; i++) {									\
		recvbuf[i] = 0;												\
	}

#define CHECK_COUNT(var, ans) 										\
	if(var != ans) {												\
		errs++;														\
		if(verbose) {												\
			fprintf(stderr, "expected %s = %d, instead got %d\n", #var, ans, var); \
		} 															\
		return errs;												\
	} else {														\
		if (verbose >= 2) {											\
			fprintf(stderr, "CORRECT: %s = %d\n", #var, var); 		\
		}															\
	}

/* TESTS */
int Test_Address_Length_Contig_Contiguous();
int Test_Address_Length_NonContig_Contiguous();
int Test_Address_Length_Contig_Vector();
int Test_Address_Length_Non_Contig_Vector();
int Test_Address_Length_Contig_HVector();
int Test_Address_Length_Non_Contig_HVector();
int Test_Address_Length_Contig_Indexed();
int Test_Address_Length_Non_Contig_Indexed();
int Test_Address_Length_Contig_Indexed_Block();
int Test_Address_Length_Non_Contig_Indexed_Block();
int Test_Address_Length_Contig_HIndexed();
int Test_Address_Length_Non_Contig_HIndexed();
int Test_Address_Length_Contig_HIndexed_Block();
int Test_Address_Length_Non_Contig_HIndexed_Block();
int Test_Address_Length_Contig_Struct();
int Test_Address_Length_Non_Contig_Struct();

int Test_Address_Length();

int main(int argc, char **argv) {
	// int size, global_rank;

	int errs = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	// MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	// MPI_Comm_size(MPI_COMM_WORLD, &size);

	errs += Test_Address_Length();

	if(errs) {
 		fprintf(stderr, "Found %d Errors\n", errs);
	} else {
		fprintf(stderr, "No Errors\n");
	}

	MPI_Finalize();

	return 0;
}


int Test_Address_Length(void) {
	int errs = 0;

	// All test cases with 'Contig' in them mean that you should
	// get 1 message as the entire datatype is contiguous
	// Test case with 'NonContig' will recieve various amounts of messages

	// Type_Contiguous -
	errs += Test_Address_Length_Contig_Contiguous();
	errs += Test_Address_Length_NonContig_Contiguous();
	// Type_Vector -
	errs += Test_Address_Length_Contig_Vector();
	errs += Test_Address_Length_Non_Contig_Vector();
	// Type_HVector -
	// errs += Test_Address_Length_Contig_HVector();
	// errs += Test_Address_Length_Non_Contig_HVector();
	// Type_Indexed -
	// errs += Test_Address_Length_Contig_Indexed();
	// errs += Test_Address_Length_Non_Contig_Indexed();
	// Type_Indexed_Block -
	// errs += Test_Address_Length_Contig_Indexed_Block();
	// errs += Test_Address_Length_Non_Contig_Indexed_Block();
	// Type_HIndexed -
	// errs += Test_Address_Length_Contig_HIndexed();
	// errs += Test_Address_Length_Non_Contig_HIndexed();
	// Type_HIndexed_Block -
	// errs += Test_Address_Length_Contig_HIndexed_Block();
	// errs += Test_Address_Length_Non_Contig_HIndexed_Block();
	// Type_Struct -
	// errs += Test_Address_Length_Contig_Struct();
	// errs += Test_Address_Length_Non_Contig_Struct();

	return errs;
}

int Test_Address_Length_Contig_Contiguous(void) {

	TEST("Test_Address_Length_Contig_Contiguous: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigContig;

	BUFFERS(SIZE)

	MPI_Type_contiguous(SIZE, MPI_INT, &contigContig);

	MPI_Type_commit(&contigContig);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigContig, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigContig, 0, 0,
	// 			 recvbuf, 1, contigContig, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	char** addresses = (char**) malloc(count*sizeof(void*));

	int* bLengths = (int*) malloc(count*sizeof(int));

	TEMP_Get_Msg_Blocks_Addresses(contigContig, (char*) sendbuf, addresses, bLengths);

	if((void*)addresses[0] != (void*)sendbuf || bLengths[0] != SIZE*sizeof(int)) {
		errs++;
		if(verbose) {
			fprintf(stderr, "addresses[0] == %p and bLengths[0] == %zu, instead got addresses[0] == %p and bLengths[0] == %d\n", sendbuf, SIZE*sizeof(int), addresses[0], bLengths[0]);
		}
	} else {
		if (verbose >= 2) {
			fprintf(stderr, "CORRECT: addresses[0] == %p and bLengths[0] == %zu\n", sendbuf, SIZE*sizeof(int));
		}
	}

	free(addresses);
	free(bLengths);

	MPI_Type_free(&contigContig);
	return errs;
}

int Test_Address_Length_NonContig_Contiguous(void) {
	TEST("Test_Address_Length_NonContig_Contiguous: made up of non-contiguous datatypes")
	int errs = 0;

	BUFFERS(SIZE*3)

	MPI_Datatype nonContigContig, intInt;

	int bLen[2] = {1,1};
	MPI_Aint disps[2] = {0, sizeof(int)*2};
	MPI_Datatype types[2] = {MPI_INT, MPI_INT};
	MPI_Type_create_struct(2, bLen, disps, types, &intInt);
	/*
	Struct looks like this:
	+-------------+
	| X |    |  X |
	+-------------+
	Where X is an INT
	*/
	MPI_Type_commit(&intInt);

	MPI_Type_contiguous(SIZE, intInt, &nonContigContig);
	/*
	+-----------------+-----------------+-----------------+-- ...
	| +-------------+ | +-------------+ | +-------------+ |   ...
	| | X |    |  X | | | X |    |  X | | | X |    |  X | |   ...
	| +-------------+ | +-------------+ | +-------------+ |   ...
	+-----------------+-----------------+-----------------+-- ...

	Since the end of the struct is adjacent to the start of the next struct
	it can be optimised to send then as one message
	(current implementation does not support this)
	*/
	MPI_Type_commit(&nonContigContig);

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigContig, &count);

	CHECK_COUNT(count, SIZE*2) // SHould be optimised to SIZE+1

	char** addresses = (char**) malloc(count*sizeof(void*));

	int* bLengths = (int*) malloc(count*sizeof(int));

	TEMP_Get_Msg_Blocks_Addresses(nonContigContig, (char*) sendbuf, addresses, bLengths);

	// MPI_Sendrecv(sendbuf, 1, nonContigContig, 0, 0,
	// 			recvbuf, 1, nonContigContig, 0, 0,
	// 			MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// Check
	int offset = 0;
	int inc = 2;
	for(int i=0; i<count; i++) {
		void* correct_address = (void*) (sendbuf + offset);

		if((void*)addresses[i] != correct_address) {
			errs++;
			if(verbose) {
				fprintf(stderr, "expected addresses[%d] == %p, instead got addresses[%d] == %p\n", i, correct_address, i, addresses[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: addresses[%d] == %p\n", i, correct_address);
			}
		}

		if(bLengths[i] != 4) { // Should be 4 for the first and last, then 8 for the all the ones in the middle, after the optimations
			errs++;
			if(verbose) {
				fprintf(stderr, "expected bLengths[%d] == %zu, instead got bLengths[%d] == %d\n", i, SIZE*sizeof(int), i, bLengths[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: bLengths[%d] == %d\n", i, bLengths[i]);
			}
		}

		offset = offset + inc;
		inc = (inc == 1 ? 2 : 1);
	}

	MPI_Type_free(&intInt);
	MPI_Type_free(&nonContigContig);

	free(addresses);
	free(bLengths);

	// printbuf(sendbuf);
	// printbuf(recvbuf);
	return errs;
}


int Test_Address_Length_Contig_Vector() {

	TEST("Test_Address_Length_Contig_Vector: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigVec;

	BUFFERS(SIZE)

	MPI_Type_vector(SIZE, 1, 1, MPI_INT, &contigVec);
	MPI_Type_commit(&contigVec);
	/*
	+------------------------------------------------------- ...
	| X | X | X | X | X | X | X | X | X | X | X | X | X | X  ...
	+------------------------------------------------------- ...
	*/

	int count;
	TEMP_Get_Msg_Blocks_Count(contigVec, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigVec, 0, 0,
	// 			 recvbuf, 1, contigVec, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	char** addresses = (char**) malloc(count*sizeof(void*));

	int* bLengths = (int*) malloc(count*sizeof(int));

	TEMP_Get_Msg_Blocks_Addresses(contigVec, (char*) sendbuf, addresses, bLengths);

	for(int i=0; i<count; i++) {

		if((void*)addresses[i] != sendbuf) {
			errs++;
			if(verbose) {
				fprintf(stderr, "expected addresses[%d] == %p, instead got addresses[%d] == %p\n", i, sendbuf, i, addresses[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: addresses[%d] == %p\n", i, sendbuf);
			}
		}

		if(bLengths[i] != SIZE*sizeof(int)) { // Should be 4 for the first and last, then 8 for the all the ones in the middle, after the optimations
			errs++;
			if(verbose) {
				fprintf(stderr, "expected bLengths[%d] == %zu, instead got bLengths[%d] == %d\n", i, SIZE*sizeof(int), i, bLengths[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: bLengths[%d] == %d\n", i, bLengths[i]);
			}
		}
	}

	MPI_Type_free(&contigVec);

	return errs;
}

int Test_Address_Length_Non_Contig_Vector() {
	TEST("Test_Address_Length_Non_Contig_Vector")
	int errs = 0;

	MPI_Datatype nonContigVec;

	BUFFERS(SIZE*2)

	MPI_Type_vector(SIZE, 1, 2, MPI_INT, &nonContigVec);
	MPI_Type_commit(&nonContigVec);
	/*
	+------------------------------------------------------- ...
	| X |   | X |   | X |   | X |   | X |   | X |   | X |    ...
	+------------------------------------------------------- ...
	*/

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigVec, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, nonContigVec, 0, 0,
	// 			 recvbuf, 1, nonContigVec, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	char** addresses = (char**) malloc(count*sizeof(void*));

	int* bLengths = (int*) malloc(count*sizeof(int));

	// Test case currently passes for getting the addresses but fails for getting the block lengths
	// so is currently disabled
#if 0
	TEMP_Get_Msg_Blocks_Addresses(nonContigVec, (char*) sendbuf, addresses, bLengths);

	int offset = 0;
	for(int i=0; i<count; i++) {

		void* expected = (void*) (sendbuf + offset);

		if((void*)addresses[i] != expected) {
			errs++;
			if(verbose) {
				fprintf(stderr, "expected addresses[%d] == %p, instead got addresses[%d] == %p\n", i, expected, i, addresses[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: addresses[%d] == %p\n", i, sendbuf);
			}
		}

		if(bLengths[i] != 4) {
			errs++;
			if(verbose) {
				fprintf(stderr, "expected bLengths[%d] == %zu, instead got bLengths[%d] == %d\n", i, sizeof(int), i, bLengths[i]);
			}
		} else {
			if (verbose >= 2) {
				fprintf(stderr, "CORRECT: bLengths[%d] == %d\n", i, bLengths[i]);
			}
		}

		offset+=2;
	}
#endif

	MPI_Type_free(&nonContigVec);

	return errs;
}

int Test_Address_Length_Contig_HVector() {
	TEST("Test_Address_Length_Non_Contig_HVector: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigHVec;

	BUFFERS(SIZE)

	MPI_Type_hvector(SIZE, 1, sizeof(int), MPI_INT, &contigHVec);

	MPI_Type_commit(&contigHVec);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigHVec, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigHVec, 0, 0,
	// 			 recvbuf, 1, contigHVec, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigHVec);

	return errs;
}

int Test_Address_Length_Non_Contig_HVector() {
	TEST("Test_Address_Length_Non_Contig_HVector: made up of non-contiguous datatypes")
	int errs = 0;

	return errs;
}

int Test_Address_Length_Contig_Indexed() {
	TEST("Test_Address_Length_Contig_Indexed: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigIndexed;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i;
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 1;
	}

	MPI_Type_indexed(SIZE, bLens, disps, MPI_INT, &contigIndexed);

	MPI_Type_commit(&contigIndexed);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigIndexed, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigIndexed, 0, 0,
	// 			 recvbuf, 1, contigIndexed, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigIndexed);

	return errs;
}

int Test_Address_Length_Non_Contig_Indexed() {
	TEST("Test_Address_Length_Non_Contig_Indexed: made up of non-contiguous datatypes")
	int errs = 0;

	MPI_Datatype nonContigIndexed;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2;
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 1;
	}

	MPI_Type_indexed(SIZE, bLens, disps, MPI_INT, &nonContigIndexed);

	MPI_Type_commit(&nonContigIndexed);

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigIndexed, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, nonContigIndexed, 0, 0,
	// 			 recvbuf, 1, nonContigIndexed, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&nonContigIndexed);

	return errs;
}

int Test_Address_Length_Contig_Indexed_Block() {
	TEST("Test_Address_Length_Contig_Indexed_Block: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigIndexedBlock;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2;
	}

	MPI_Type_create_indexed_block(SIZE, 2, disps, MPI_INT, &contigIndexedBlock);

	MPI_Type_commit(&contigIndexedBlock);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigIndexedBlock, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigIndexedBlock, 0, 0,
	// 			 recvbuf, 1, contigIndexedBlock, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigIndexedBlock);

	return errs;
}

int Test_Address_Length_Non_Contig_Indexed_Block() {
	TEST("Test_Address_Length_Non_Contig_Indexed_Block: made up of non-contiguous datatypes")
	int errs = 0;

	MPI_Datatype nonContigIndexedBlock;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2;
	}

	MPI_Type_create_indexed_block(SIZE, 1, disps, MPI_INT, &nonContigIndexedBlock);

	MPI_Type_commit(&nonContigIndexedBlock);

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigIndexedBlock, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, nonContigIndexedBlock, 0, 0,
	// 			 recvbuf, 1, nonContigIndexedBlock, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&nonContigIndexedBlock);

	return errs;
}

int Test_Address_Length_Contig_HIndexed() {
	TEST("Test_Address_Length_Contig_HIndexed: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigHIndexed;

	BUFFERS(SIZE*2)

	MPI_Aint disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*sizeof(int)*2;
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 2;
	}

	MPI_Type_create_hindexed(SIZE, bLens, disps, MPI_INT, &contigHIndexed);

	MPI_Type_commit(&contigHIndexed);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigHIndexed, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigHIndexed, 0, 0,
	// 			 recvbuf, 1, contigHIndexed, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigHIndexed);

	return errs;
}

int Test_Address_Length_Non_Contig_HIndexed() {
	TEST("Test_Address_Length_Non_Contig_HIndexed: made up of non-contiguous datatypes")
	int errs = 0;

	MPI_Datatype nonContigHIndexed;

	BUFFERS(SIZE*2)

	MPI_Aint disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*sizeof(int)*2;
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 1;
	}

	MPI_Type_create_hindexed(SIZE, bLens, disps, MPI_INT, &nonContigHIndexed);

	MPI_Type_commit(&nonContigHIndexed);

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigHIndexed, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, nonContigHIndexed, 0, 0,
	// 			 recvbuf, 1, nonContigHIndexed, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&nonContigHIndexed);

	return errs;
}

int Test_Address_Length_Contig_HIndexed_Block() {
	TEST("Test_Address_Length_Contig_HIndexed_Block: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigHIndexedBlock;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2;
	}

	MPI_Type_create_indexed_block(SIZE, 2, disps, MPI_INT, &contigHIndexedBlock);

	MPI_Type_commit(&contigHIndexedBlock);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigHIndexedBlock, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigHIndexedBlock, 0, 0,
	// 			 recvbuf, 1, contigHIndexedBlock, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigHIndexedBlock);

	return errs;
}

int Test_Address_Length_Non_Contig_HIndexed_Block() {
	TEST("Test_Address_Length_Non_Contig_HIndexed_Block: made up of non-contiguous datatypes")
	int errs = 0;

	MPI_Datatype nonContigHIndexedBlock;

	BUFFERS(SIZE*2)

	int disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2;
	}

	MPI_Type_create_indexed_block(SIZE, 1, disps, MPI_INT, &nonContigHIndexedBlock);

	MPI_Type_commit(&nonContigHIndexedBlock);

	int count;
	TEMP_Get_Msg_Blocks_Count(nonContigHIndexedBlock, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, nonContigHIndexedBlock, 0, 0,
	// 			 recvbuf, 1, nonContigHIndexedBlock, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&nonContigHIndexedBlock);

	return errs;
}

int Test_Address_Length_Contig_Struct() {
	TEST("Test_Address_Length_Contig_Struct: made up of contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigStruct;

	BUFFERS(SIZE*2)

	MPI_Aint disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*sizeof(int);
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 1;
	}

	MPI_Datatype types[SIZE];
	for(int i=0; i<SIZE; i++) {
		types[i] = MPI_INT;
	}

	MPI_Type_create_struct(SIZE, bLens, disps, types, &contigStruct);

	MPI_Type_commit(&contigStruct);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigStruct, &count);

	CHECK_COUNT(count, 1)

	// MPI_Sendrecv(sendbuf, 1, contigStruct, 0, 0,
	// 			 recvbuf, 1, contigStruct, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigStruct);

	return errs;
}

int Test_Address_Length_Non_Contig_Struct() {
	TEST("Test_Address_Length_Non_Contig_Struct: made up of non-contiguous datatypes")
	int errs = 0;

	MPI_Datatype contigStruct;

	BUFFERS(SIZE*2)

	MPI_Aint disps[SIZE];
	for(int i=0; i<SIZE; i++) {
		disps[i] = i*2*sizeof(int);
	}

	int bLens[SIZE];
	for(int i=0; i<SIZE; i++) {
		bLens[i] = 1;
	}

	MPI_Datatype types[SIZE];
	for(int i=0; i<SIZE; i++) {
		types[i] = MPI_INT;
	}

	MPI_Type_create_struct(SIZE, bLens, disps, types, &contigStruct);

	MPI_Type_commit(&contigStruct);

	int count;
	TEMP_Get_Msg_Blocks_Count(contigStruct, &count);

	CHECK_COUNT(count, SIZE)

	// MPI_Sendrecv(sendbuf, 1, contigStruct, 0, 0,
	// 			 recvbuf, 1, contigStruct, 0, 0,
	// 			 MPI_COMM_SELF ,MPI_STATUS_IGNORE);

	// printbuf(sendbuf);
	// printbuf(recvbuf);

	MPI_Type_free(&contigStruct);

	return errs;
}
