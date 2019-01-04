/**
 * \addtogroup CkLdb
*/
/*@{*/


#include <stdio.h>
#include <stdlib.h>
#include "bitvecset.h"

BV_Set * makeSet(int *list, int size, int max);

BV_Set * makeEmptySet(int max)
{
 int x;

 return ( makeSet(&x, 0, max));
  
}


BV_Set * makeSet(int *list, int size, int max)
{
  BV_Set * s;
  int i;

  s = (BV_Set *) malloc(sizeof(BV_Set));
  s->max = max;
  s->size = size;
  s->vector = (short *) malloc(sizeof(short)*(1+max));

  for (i = 0; i<= max; i++)
      s->vector[i] = 0;
  for (i = 0; i< size; i++)
    if (list[i] <= max && list[i] >= 0)
      s->vector[ list[i]] = 1;
    else 
      printf("***ERROR: makeSet received %d, when max was supposed to be %d",
	     list[i], max);

  return s;
}

void destroySet(BV_Set* set)
{
  free(set->vector);
  set->vector=0;
  free(set);
}


void bvset_insert(BV_Set * s, int value)
{
  if (value > s->max || value < 0) 
    printf("BV_Set error. inserting value %d in a set where max is %d\n",
	   value, s->max);
  else {
    if (s->vector[value]) return;
    else { 
      s->vector[value] = 1;
      s->size++;
    }
  }
}


int bvset_find(BV_Set * s, int value)
{
  if (value > s->max || value < 0) {
    printf("BV_Set error.  *find* on a value %d in a set where max is %d\n",
	   value, s->max);
    return -1;
  }
  else return (s->vector[value]);
}

int bvset_size(BV_Set * s) 
{
  return s->size;
}

void bvset_enumerate(BV_Set * s, int **list, int *size)
{
  int i, j;

  /*
  printf("set is: ");
  for (i=0; i<=s->max; i++)
    printf("%d ", s->vector[i]);
  printf("\n returning list: "); */

  *list = (int *) malloc(sizeof(int)*s->size);
  *size = s->size;

  j = 0;
  for (i=0; i<=s->max; i++)
    if (s->vector[i]) (*list)[j++] = i;

  if (j > s->size) {
    printf("Error, too many bits written %d %d\n",j,s->size);
    printf("set is: ");
    for (i=0; i<=s->max; i++)
      printf("%d ", s->vector[i]);
    printf("\n returning list: ");
    for (i=0; i< *size; i++)
      printf("%d ", (*list)[i]);
  }
}

/*@}*/
