#include "tm_timings.h"

static CLOCK_T time_tab[MAX_CLOCK];
static int clock_num=-1;

void get_time(){
  clock_num++;

  if(clock_num>MAX_CLOCK-1)
    return;

  
  CLOCK(time_tab[clock_num]);
}

double time_diff(){
  CLOCK_T t2;
  
  if(clock_num>MAX_CLOCK-1){
    clock_num--;
    return -1.0;
  }

  CLOCK(t2);

  return CLOCK_DIFF(t2,time_tab[clock_num--]);
}
