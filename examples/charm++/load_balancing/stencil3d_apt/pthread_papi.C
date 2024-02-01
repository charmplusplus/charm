#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "papi.h"

static unsigned char *memory;

struct thread_data_t {
  int thread_number;
  unsigned char *offset;
  int size;
};


void *init_memory(void *data) {

  struct thread_data_t *t;
  int i;
  int eventset=PAPI_NULL;
  long long values[2];
  int result;

  t = (struct thread_data_t *)data;

  result = PAPI_register_thread(  );
  if ( result != PAPI_OK ) {
    printf("Error register thread!\n");
  }


  result=PAPI_create_eventset(&eventset);
  if (result!=PAPI_OK) {
    printf("Error PAPI create eventset: %s\n",PAPI_strerror(result));
  }

  result=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
  if (result!=PAPI_OK) {
    printf("Error PAPI add_event %s\n",
      PAPI_strerror(result));
  }
  result=PAPI_add_named_event(eventset,"PAPI_L3_TCM");
  if (result!=PAPI_OK) {
    printf("Error PAPI add_event %s\n",
      PAPI_strerror(result));
  }

  PAPI_start(eventset);

  printf("\tIn thread #%d initializing %d bytes at %p\n",
    t->thread_number,
    t->size,
    t->offset);

  for(i=0;i < t->size; i++) {
    t->offset[i]=0xa5;
  }
  printf("\tFinished %d\n",t->thread_number);

  PAPI_stop(eventset,values);

  printf("Total instructions %lld, L3 misses %lld\n",values[0],values[1]);

  pthread_exit(NULL);
}


int main (int argc, char **argv) {

  int num_threads=1;
  int mem_size=256*1024*1024; /* 256 MB */
  pthread_t *threads;
  struct thread_data_t *thread_data;
  int result;
  long int t;
  int i,retval;

  retval=PAPI_library_init(PAPI_VER_CURRENT);
  if (retval!=PAPI_VER_CURRENT) {
    fprintf(stderr,"Error initializing PAPI!\n");
  }

  if (PAPI_thread_init(pthread_self) != PAPI_OK) {
    perror("PAPI_thread_init");
  }

  /* Set number of threads from the command line */
  if (argc>1) {
    num_threads=atoi(argv[1]);
  }

  /* allocate memory */
  memory=(unsigned char *)malloc(mem_size);
  if (memory==NULL) perror("allocating memory");

  /* allocate threads */
  threads=(pthread_t*)calloc(num_threads, sizeof(pthread_t));
  if (threads==NULL) perror("allocating threads");

  /* allocate thread data */
  /* Why must we have unique thread datas? */
  thread_data=(thread_data_t*)calloc(num_threads,sizeof(struct thread_data_t));
  if (thread_data==NULL) perror("allocating thread_data");

  printf("Initializing %d MB of memory using %d threads\n",
    mem_size/(1024*1024),num_threads);

  for(t=0; t<num_threads; t++) {

    thread_data[t].thread_number=t;
    thread_data[t].size=mem_size/num_threads;
    thread_data[t].offset=memory+(t*thread_data->size);

    printf("Creating thread %ld\n", t);
    result = pthread_create(
      &threads[t],
      NULL,
      init_memory,
      (void *)&thread_data[t]);

    if (result) {
      fprintf(stderr,"ERROR: pthread_create returned %d\n",
          result);
      exit(-1);
    }

//    pthread_join(threads[t],NULL);

  }

  for(t=0;t<num_threads;t++) {
    pthread_join(threads[t],NULL);
  }

  printf("Master thread exiting, validating results\n");

  for(i=0;i<mem_size;i++) {
    if (memory[i]!=0xa5) printf("Uninitialized byte at %d %x\n",
      i,memory[i]);
  }

  PAPI_shutdown();

  pthread_exit(NULL);
}

