#ifndef _CString_H_
#define _CString_H_

#include <string.h>
#include <stdio.h>
#include "sdag-globals.h"

class CString {
  private:
    unsigned int len;
    unsigned int stringlen;
    char *text;
  public:
    CString(int initlen=128)
    {
      len = initlen;
      stringlen = 0;
      text = new char[initlen];
      text[0] = '\0';
    }
    CString(const char *txt)
    {
      stringlen = strlen(txt);
      len = (stringlen+1)*2;
      text = new char[len];
      strcpy(text, txt);
    }
    CString(CString *str)
    {
      stringlen = str->length();
      len = (stringlen+1)*2;
      text = new char[len];
      strcpy(text, str->charstar());
    }
    ~CString(){
      // delete[] text;
    }
    void append(char *txt)
    {
      stringlen += strlen(txt);
      while (stringlen>=len)
        len *= 2;
      char *newstr = new char[len];
      strcpy(newstr, text);
      strcat(newstr, txt);
      delete[] text;
      text = newstr;
    }
    void append(CString* cstr)
    {
      char *txt = cstr->charstar();
      stringlen += cstr->length();
      while (stringlen>=len)
        len *= 2;
      char *newstr = new char[len];
      strcpy(newstr, text);
      strcat(newstr, txt);
      delete[] text;
      text = newstr;
    }
    unsigned int length(void)
    {
      return stringlen;
    }
    char *charstar(void)
    {
      return text;
    }
    void print(int indent) {
      Indent(indent);
      printf("%s", text);
    }
    int equal(CString *another) {
      return (strcmp(text, another->text)==0) ? 1 : 0;
    }
};

#endif /* _CString_H_ */
