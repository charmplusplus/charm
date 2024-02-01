#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

main()
{
int retval, EventSet = PAPI_NULL;
long_long values[3];
    unsigned counter;
    unsigned c;
    unsigned long fact;
    unsigned stoppoint;


        /* Initialize the PAPI library */
        retval = PAPI_library_init(PAPI_VER_CURRENT);

        if (retval != PAPI_VER_CURRENT) {
          fprintf(stderr, "PAPI library init error!\n");
          exit(1);
        }

        /* Create the Event Set */
        if (PAPI_create_eventset(&EventSet) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);

        /* Add Total Instructions Executed to our EventSet */
        if (PAPI_add_event(EventSet, PAPI_TOT_INS) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);

        /* Add Total Instructions Executed to our EventSet */
        if (PAPI_add_event(EventSet, PAPI_TOT_CYC) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);

        /* Add Total Instructions Executed to our EventSet */
        if (PAPI_add_event(EventSet, PAPI_LST_INS) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);


   srand ( time(NULL) );
   stoppoint = 50+(rand() % 100);
/* Do some computation here */
    for (counter = 0; counter < stoppoint; counter++)
    {
        /* Initialize the PAPI library */
        retval = PAPI_library_init(PAPI_VER_CURRENT);

        if (retval != PAPI_VER_CURRENT) {
          fprintf(stderr, "PAPI library init error!\n");
          exit(1);
        }


        /* Start counting */
        if (PAPI_start(EventSet) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);


                fact = 1;
        for (c = 1; c <= counter; c++)
        {
                     fact = c * c;
        }




        printf("Factorial of %d is %lu\n", counter, fact);
        if (PAPI_read(EventSet, values) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);
        printf ("\nTotal Instructions Completed:\t%lld\t Total Cycles:\t%lld\tLoad Store Instructions\t%lld\n\n", values[0], values[1], values[2]);
        /* Do some computation here */

        if (PAPI_stop(EventSet, values) != PAPI_OK)
            printf ("%s:%d\t ERROR\n", __FILE__, __LINE__);

            }
        /* End of computation */


}
