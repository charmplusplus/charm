
#ifndef _CKLISTSTRING_H
#define _CKLISTSTRING_H

// a simple class which maintain a list of numbers in such format:
//  0-10,20-40:2,100,200
//  no space is allowed

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for strtok

class CkListString
{
private:
  char *list;
public:
  CkListString(): list(NULL) {}
  CkListString(char *s) { list = strdup(s); }
  ~CkListString() { if (list) free(list); }
  void set(char *s) { list = strdup(s); }
  int isEmpty() { return list == NULL; }
  int includes(int p) {
    size_t i; 
    int ret = 0;
    if (list == NULL) return 1;    // subtle: empty string means ALL
    char *dupstr = strdup(list);   // don't touch the orignal string
    char *str = strtok(dupstr, ",");
    while (str) {
      int hasdash=0, hascolon=0, hasdot=0;
      for (i=0; i<strlen(str); i++) {
          if (str[i] == '-') hasdash=1;
          if (str[i] == ':') hascolon=1;
          if (str[i] == '.') hasdot=1;
      }
      int start, end, stride=1, block=1;
      if (hasdash) {
          if (hascolon) {
            if (hasdot) {
              if (sscanf(str, "%d-%d:%d.%d", &start, &end, &stride, &block) != 4)
                 printf("Warning: Check the format of \"%s\".\n", str);
            }
            else {
              if (sscanf(str, "%d-%d:%d", &start, &end, &stride) != 3)
                 printf("Warning: Check the format of \"%s\".\n", str);
            }
          }
          else {
            if (sscanf(str, "%d-%d", &start, &end) != 2)
                 printf("Warning: Check the format of \"%s\".\n", str);
          }
      }
      else {
          sscanf(str, "%d", &start);
          end = start;
      }
      if (block > stride) {
        printf("Warning: invalid block size in \"%s\" ignored.\n", str);
        block=1;
      }
      //CmiPrintf("GOT %d %d %d %d\n", start, end, block, stride);
      if (p<=end && p>=start) {
          if ((p-start)%stride < block) ret = 1;
          break;
      }
      str = strtok(NULL, ",");
    }
    free(dupstr);
    return ret;
  }
};

#endif
