
#ifndef _CKLISTSTRING_H
#define _CKLISTSTRING_H

// a simple class which maintain a list of numbers in such format:
//  0-10,20-40,100,200

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for strtok

class CkListString
{
private:
  char *list;
public:
  CkListString(): list(NULL) {}
  CkListString(char *s): list(s) {}
  ~CkListString() { if (list) free(list); }
  void set(char *s) { list = s; }
  int isEmpty() { return list == NULL; }
  int includes(int p) {
    size_t i; 
    int ret = 0;
    char *dupstr = strdup(list);   // don't touch the orignal string
    char *str = strtok(dupstr, ",");
    while (str) {
      for (i=0; i<strlen(str); i++)
          if (str[i] == '-') break;
      int start, end;
      if (i<strlen(str))
          sscanf(str, "%d-%d", &start, &end);
      else {
          sscanf(str, "%d", &start);
          end = start;
      }
      if (p<=end && p>=start) {
          ret = 1;
          break;
      }
      str = strtok(NULL, ",");
    }
    free(dupstr);
    return ret;
  }
};

#endif
