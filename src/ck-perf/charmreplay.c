/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.1  1998-02-27 11:52:49  jyelon
 * Cleaned up header files, replaced load-balancer.
 *
 * Revision 2.0  1995/06/02 17:40:29  brunner
 * Reorganized directory structure
 *
 * Revision 1.1  1994/10/14  21:00:47  brunner
 * Initial revision
 *
 ***************************************************************************/

#include <sys/param.h>
#include <stdio.h>
#include <string.h>

#define GOOD 0

#define DEBUG 1
#define PROJECTIONS 2

#define MAXLENGTH 1024

/** Needs to be changed in altered in charm.h **/
#define NewChareMsg             0
#define ForChareMsg             1
#define BocInitMsg              2
#define BocMsg                  3
#define TerminateToZero         4
#define TerminateSys            5
#define InitCountMsg            6
#define ReadVarMsg              7
#define ReadMsgMsg              8
#define BroadcastBocMsg         9
#define DynamicBocInitMsg       10
#define LdbMsg                  12
#define VidMsg                  13
#define QdBocMsg                14
#define QdBroadcastBocMsg       15
#define AccInitMsg              21
#define MonoInitMsg             22



#define  CREATION           1
#define  BEGIN_PROCESSING   2
#define  END_PROCESSING     3
#define  ENQUEUE            4
#define  DEQUEUE            5
#define  BEGIN_COMPUTATION  6
#define  END_COMPUTATION    7
#define  BEGIN_INTERRUPT    8
#define  END_INTERRUPT      9
#define  INSERT             10
#define  DELETE             11
#define  FIND               12



#define HASH_TABLE_SIZE 2591
#define HashMap(a, b) ((20011*a+20021*b) % HASH_TABLE_SIZE)

typedef struct entry {
	int pe, event, destination; 
	struct entry *next;
} ENTRY;

ENTRY *hash_table[HASH_TABLE_SIZE];

int number_pe;
char *filename, *pwd;
int TotalChares, TotalEps, TotalMsgs, TotalPseudos;

InsertDestination(pe, event, destination)
int pe, event, destination;
{
	ENTRY *current;
	int index = HashMap(pe, event);

	current = (ENTRY *) malloc(sizeof(ENTRY));
	current->pe = pe;
	current->event = event;
	current->destination = destination;
	current->next = hash_table[index];
	hash_table[index] = current;
}



FindDestination(event, pe)
int event, pe;
{
	int index = HashMap(pe, event);
	ENTRY *current = hash_table[index];

	while (current != 0)
		if (current->pe==pe && current->event==event)
			return current->destination;
		else
			current = current->next;
	printf("*** ERROR *** Cannot determine destination for %d, %d\n",
				pe, event);
	return -1;
}



main(argc, argv)
int argc;
char *argv[];
{ 
	int	i, j;
	FILE *fp;
	int mode = -1;
	char template[1000];
	char *getcwd(), *mktemp();
	char name[MAXLENGTH], what[MAXLENGTH], pathname[MAXLENGTH];

	filename = argv[1];
	read_in_state_file(filename);
	strcpy(template,  ".tempXXXXXX");

	if ((pwd = getcwd(pathname, MAXLENGTH)) == 0) printf("ERROR: %s\n", pathname);

	for (i=0; i<HASH_TABLE_SIZE; i++)
		hash_table[i] = 0;

	for (i=0; i<number_pe; i++)
	{
		sprintf(name, "%s/%s.%d.log", pathname, filename, i);
		fp = fopen(name, "r");
		if (fp == 0) {
			printf("*** ERROR *** Unable to open log file %s\n", name);
			return GOOD;
		}
		fscanf(fp, "%s", what);
		if (!strcmp(what, "DEBUG-REPLAY") || 
			!strcmp(what, "PROJECTIONS-REPLAY")) {
				fclose(fp);
				return GOOD;
		}
		if (!strcmp(what, "DEBUG-RECORD")) 
			mode = DEBUG;
		else if (!strcmp(what, "PROJECTIONS-RECORD")) 
			mode = PROJECTIONS;
		else 
			printf("*** ERROR *** Unknown type of log file %s\n", name);
		
		if (mode==DEBUG) read_in_debug_file(fp, i);
		else if (mode==PROJECTIONS) read_in_projections_file(fp, i);
		fclose(fp);
	}

	mktemp(template);
	for (i=0; i<number_pe; i++)
	{
		FILE	*fp1, *fp2;
		char 	command[MAXLENGTH];

		sprintf(name, "%s/%s.%d.log", pathname, filename, i);
		fp1 = fopen(name, "r");
		if (fp1 == 0) {
			printf("*** ERROR *** Unable to open log file %s\n", name);
			return GOOD;
		}

		fp2 = fopen(template, "w");
		if (fp2 == 0) {
			printf("*** ERROR *** Unable to open log file %s\n", 
					template);
			return GOOD;
		}
		
		if (mode==DEBUG) write_out_debug_file(fp1, fp2, i);
		else if (mode==PROJECTIONS) write_out_projections_file(fp1, fp2, i);
		fclose(fp1);
		fclose(fp2);

		sprintf(name, "%s/%s.%d.rpy", pathname, filename, i);
		sprintf(command, "cp %s %s", template, name);
		system(command);
	}
	return GOOD;
}


/*************************************************************************/
/** Read in state file information.										**/
/*************************************************************************/
read_in_state_file(filename)
char *filename;
{
	FILE *fp;
	int done;
	int id, chareid,msgid;
	char type[1000], name[1000];
	int size, msg_index, pseudo_index, pseudo_type;

	/*****************************************************************/
	/** Get the file name and open it.              **/
	/*****************************************************************/
	sprintf(name, "%s.sts", filename);
	fp = fopen(name, "r");
	if (fp == 0)
		printf("*** ERROR *** Unable to open log file %s\n", name);
	done = 0;
	while (!done)
	{
		fscanf(fp, "%s", type);
		if (!strcmp(type, "ENTRY"))
			fscanf(fp, "%d %s %d %d", &id, name, &chareid, &msgid);
		else if (!strcmp(type, "CHARE") || (!strcmp(type, "BOC")))
			fscanf(fp, "%d %s", &id, name);
		else if (!strcmp(type, "MACHINE"))
			fscanf(fp, "%s", name);
		else if (!strcmp(type, "PROCESSORS"))	
			fscanf(fp, "%d", &number_pe);
		else if (!strcmp(type, "MESSAGE"))	
			fscanf(fp, "%d %d", &msg_index, &size);
		else if (!strcmp(type, "PSEUDO"))	
			fscanf(fp, "%d %d %s", &pseudo_index, &pseudo_type, name);
		else if (!strcmp(type, "TOTAL_CHARES"))	
			fscanf(fp, "%d", &TotalChares);
		else if (!strcmp(type, "TOTAL_EPS"))	
			fscanf(fp, "%d", &TotalEps);
		else if (!strcmp(type, "TOTAL_MSGS"))	
			fscanf(fp, "%d", &TotalMsgs);
		else if (!strcmp(type, "TOTAL_PSEUDOS"))	
			fscanf(fp, "%d", &TotalPseudos);
		else if (!strcmp(type, "END"))
			done = 1;
	}
}


/*************************************************************************/
/** This function is used to read in a log file and generate the display**/
/** information for it.							**/
/*************************************************************************/

read_in_debug_file(fp, me)
FILE *fp;
int me;
{
	int i;
	unsigned int time;
	int type, mtype, entry, event, dest, pe;

	/*********************************************************/
	/** Read in the entries and process them.		**/
	/*********************************************************/
	while (read_in_debug_line(fp, 0, &type, &mtype, &entry, &time, &event, 
								&dest, &pe) != EOF) {
		/*************************************************/
		/** Perform appropriate actions for this entry.	**/
		/*************************************************/
		switch (type)
		{
		case BEGIN_PROCESSING:
			if (mtype==NewChareMsg) InsertDestination(pe, event, me);
			break;

		case END_COMPUTATION: 
			break;
		}
	}
}


write_out_debug_file(fp1, fp2, me)
	FILE *fp1, *fp2;
	int me;
{
	int i;
	char what[100];
	unsigned int time;
	int type, mtype, entry, event, pe, dest;

	fscanf(fp1, "%s", what);
	fprintf(fp2, "DEBUG-REPLAY\n");

	/*********************************************************/
	/** Read in the entries and process them.		**/
	/*********************************************************/
	while (read_in_debug_line(fp1, 0, &type, &mtype, &entry, &time, &event, 
								&dest, &pe) != EOF) {

		write_out_debug_line(fp2, type, mtype, entry, time, event, pe);
		switch (type) {

		case CREATION:
			if (mtype==NewChareMsg) 
				fprintf(fp2, " %d", FindDestination(event, me));
			break;

		case END_COMPUTATION:
			break;
		}
		fprintf(fp2, "\n");
	}
}


/*************************************************************************/
/** This function is used to read in a log file and generate the display**/
/** information for it.							**/
/*************************************************************************/

read_in_projections_file(fp, me)
	FILE *fp;
	int me;
{
	unsigned int time;
	int type, mtype, entry, event, pe, dest;

	/*********************************************************/
	/** Read in the entries and process them.		**/
	/*********************************************************/
	while (read_in_projections_data(fp, 0, &type, &mtype, &entry, &time, 
									&event, &dest, &pe) != EOF) {
		/*************************************************/
		/** Perform appropriate actions for this entry.	**/
		/*************************************************/
		switch (type)
		{
		case BEGIN_PROCESSING:
			if (mtype == NewChareMsg) InsertDestination(pe, event, me);
			break;

		case END_COMPUTATION:
			break;
		}
	}
}

write_out_projections_file(fp1, fp2, me)
	FILE *fp1, *fp2;
	int me;
{
	char what[100];
	unsigned int time;
	int type, mtype, entry, event, pe, dest;


	fscanf(fp1, "%s", what);
	fprintf(fp2, "PROJECTIONS-REPLAY\n");

	/*********************************************************/
	/** Read in the entries and process them.		**/
	/*********************************************************/
	while (read_in_projections_data(fp1, 0, &type, &mtype, &entry, &time, 
									&event, &dest, &pe) != EOF) {
		write_out_projections_line(fp2, type, mtype, entry, time, event, pe);

		/*************************************************/
		/** Perform appropriate actions for this entry.	**/
		/*************************************************/
		switch (type)
		{
		case CREATION:
			if (mtype==NewChareMsg)
				fprintf(fp2, " %d", FindDestination(event, pe));
			break;

		case END_COMPUTATION:
			break;

		}
		fprintf(fp2, "\n");
	}
}
