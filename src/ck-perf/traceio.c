#include <stdio.h>
#include "chare.h"
#include "globals.h"
#include "trace.h"

read_in_debug_line(fp, replay, type, mtype, entry, time, event, dest, pe)
FILE *fp;
int replay;
int *type, *mtype, *entry;
unsigned int *time;
int *event, *dest,  *pe;
{
	int value;

	value = fscanf(fp, "%d", type);
	if (value==EOF) return value;

	switch (*type)
	{
		case CREATION:
			value = fscanf(fp, "%d %d %d\n", mtype, entry, event);
			if (value==EOF) return value;
			if (replay && *mtype==NewChareMsg) 
				value = fscanf(fp, "%d", dest);
			return value;

		case BEGIN_PROCESSING:
			return fscanf(fp, "%d %d %d %d\n", mtype, entry, event, pe);

		case END_COMPUTATION:
			return value;

		default:
			printf("***ERROR*** Wierd Event %d.\n", type);
			return value; 
	}
}


void write_out_debug_line(fp, type, mtype, entry, time, event, pe)
FILE *fp;
int type, mtype, entry;
unsigned int time;
int event, pe;
{
	fprintf(fp, "%d ", type);

	switch (type)
	{
		case CREATION:
			fprintf(fp, "%d %d %d", mtype, entry, event);
			break;

		case BEGIN_PROCESSING:
			fprintf(fp, "%d %d %d %d", mtype, entry, event, pe);
			break;

		case END_COMPUTATION:
			break;

		default:
			printf("***ERROR*** Wierd Event %d.\n", type);
			break;
	}
}


read_in_projections_data(fp, replay, type, mtype, entry, time, event, dest, pe)
FILE *fp;
int replay;
int *type, *mtype, *entry;
unsigned int *time;
int *event, *dest, *pe;
{
	int value;

   	value = fscanf(fp, "%d", type);
	if (value==EOF) return value;

   	switch (*type) {
       	case CREATION:
       	case END_PROCESSING:
	 	case BEGIN_PROCESSING:
          	value = fscanf(fp, "%d %d %u %d %d", mtype, entry, time, event, pe);
			if (value==EOF) return value;
			if (replay && *type==CREATION && *mtype==NewChareMsg) 
				value = fscanf(fp, "%d", dest);
			return value;

       	case ENQUEUE:
       	case DEQUEUE:
           	return fscanf(fp, "%d %u %d %d", mtype, time, event, pe);
	
       	case INSERT:
       	case FIND:
       	case DELETE:
           	return fscanf(fp, "%d %d %u %d", mtype, entry, time, pe);

       	case BEGIN_INTERRUPT:
       	case END_INTERRUPT:
           	return fscanf(fp, "%u %d %d", time, event, pe);
	
       	case BEGIN_COMPUTATION:
       	case END_COMPUTATION:
           	return fscanf(fp, "%u", &time);
	
       	default:
           	printf("***ERROR*** Wierd Event %d.\n", *type);
			return value;
    }
}

void write_out_projections_line(fp, type, mtype, entry, time, event, pe)
FILE *fp;
int type, mtype, entry;
unsigned int time;
int event, pe;
{
	fprintf(fp, "%d ", type);

	/*************************************************/
	/** Perform appropriate actions for this entry.	**/
	/*************************************************/
	switch (type)
	{
        case USER_EVENT:
                fprintf(fp, "%d %u %d %d", mtype, time, event, pe);
                break;

        case BEGIN_IDLE:
        case END_IDLE:
        case BEGIN_PACK:
        case END_PACK:
        case BEGIN_UNPACK:
        case END_UNPACK:
                fprintf(fp, "%u %d", time, pe);
                break;

	case CREATION:
	case BEGIN_PROCESSING:
	case END_PROCESSING:
		fprintf(fp, "%d %d %u %d %d", mtype, entry, time, event, pe);
		break;

	case ENQUEUE:
	case DEQUEUE:
		fprintf(fp, "%d %u %d %d", mtype, time, event, pe);
		break;

	case INSERT:
	case FIND:
	case DELETE:
		fprintf(fp, "%d %d %u %d", mtype, entry, time, pe);
		break;

	case BEGIN_INTERRUPT:
	case END_INTERRUPT:
		fprintf(fp, "%u %d %d", time, event, pe);
		break;

	case BEGIN_COMPUTATION:
	case END_COMPUTATION:
    	fprintf(fp, "%u", time);
		break;

	default:
		printf("***ERROR*** Wierd Event %d.\n", type);
		break;
	}
}


