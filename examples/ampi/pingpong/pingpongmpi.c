#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <mpi.h>

#define INIT_SEC 1000000

struct itimerval *tim;
unsigned int Timers = 0;

void   Create_Timers (int n);
void   Start_Timer   (int i, int which);
float  Read_Timer    (int i, int which);

void Create_Timers (int n)
    {
    if( Timers > 0 )
        {
        fprintf (stderr, "Create_Timers: timers already created!\n");
        exit (-1);
        }
    
    tim = (struct itimerval*) malloc (n * sizeof (struct itimerval));
    Timers = n;
    }

void Start_Timer (int i, int which)
    {
    if( i >= Timers )
        {
        fprintf (stderr, "Start_Timers: out-of-range timer index %d\n", i);
        exit (-1);
        }
    
    tim[i].it_value.tv_sec = INIT_SEC;
    tim[i].it_value.tv_usec = 0;
    tim[i].it_interval.tv_sec = INIT_SEC;
    tim[i].it_interval.tv_usec = 0;
    
    setitimer (which, &(tim[i]), NULL);
    }

float Read_Timer (int i, int which)
    {
    float elapsed_time;

    if( i >= Timers )
        {
        fprintf (stderr, "Read_Timer: out-of-range timer index %d\n", i);
        exit (-1);
        }
    
    getitimer (which, &(tim[i]));

    elapsed_time = ( (float)INIT_SEC - tim[i].it_value.tv_sec ) -
	           ( (float)tim[i].it_value.tv_usec/1000000 );

    return elapsed_time;
    }

int main(int argc, char **argv)
    {
    int my_id;		/* process id */
    int p;		/* number of processes */
    char* message;	/* storage for the message */
    int i, k, max_msgs, msg_size;
    MPI_Status status;	/* return status for receive */
    float elapsed_time_sec;
    float bandwidth;
    int nreceives = 0;
    int ncount = 0;
    MPI_Request request[256];

    printf("Starting benchmark\n\n");

    MPI_Init( &argc, &argv );
    
    MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
    MPI_Comm_size( MPI_COMM_WORLD, &p );

    if (argc < 3)
	{
	fprintf (stderr, "need iterations and msg size as params\n");
	goto EXIT;
	}

    if ((sscanf (argv[1], "%d", &max_msgs) < 1) ||
        		(sscanf (argv[2], "%d", &msg_size) < 1))
	{
	fprintf (stderr, "need msg count and msg size as params\n");
	goto EXIT;
        
	}

    if(argc > 3)
        nreceives = atoi(argv[3]);

    message = (char*)malloc (msg_size);

    for(ncount = 0; ncount < nreceives; ncount ++)
        MPI_Irecv(message, msg_size, MPI_CHAR, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, &request
                  [ncount]);

    /* don't start timer until everybody is ok */
    MPI_Barrier(MPI_COMM_WORLD); 

    if( my_id < p/2 )
        {
	Create_Timers (1);
	Start_Timer (0, ITIMER_REAL);

	for(i=0; i<max_msgs; i++){
	    MPI_Send(message, msg_size, MPI_CHAR, my_id+p/2, 0, MPI_COMM_WORLD);
	    MPI_Recv(message, msg_size, MPI_CHAR, my_id+p/2, 0, MPI_COMM_WORLD, &status); 
        }

        MPI_Barrier(MPI_COMM_WORLD); 

	elapsed_time_sec = Read_Timer (0, ITIMER_REAL) * 1000.0 * 1000 / 2 / max_msgs;
	bandwidth = 8 * msg_size / elapsed_time_sec;

	fprintf (stdout, "%5d %7d\t ", max_msgs, msg_size);
	fprintf (stdout,"%8.3f us,\t %8.3f Mbits/sec\n",
	    elapsed_time_sec, bandwidth);

        }
    else
        {
	for(i=0; i<max_msgs; i++){
            MPI_Recv(message, msg_size, MPI_CHAR, my_id-p/2, 0, MPI_COMM_WORLD, &status); 
            MPI_Send(message, msg_size, MPI_CHAR, my_id-p/2, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD); 
        }	    

    free(message);
    free(tim);
EXIT:
    MPI_Finalize();

    return 0;
}

