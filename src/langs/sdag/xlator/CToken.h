#ifndef _CToken_H_
#define _CToken_H_

#include <stdio.h>
#include "EToken.h"
#include "CString.h"
#include "sdag-globals.h"

class CToken {
  public:
    EToken type;
    CString *text;
    CToken(EToken t, const char *txt)
    {
      type = t;
      text = new CString(txt);
    }
    ~CToken()
    {
      // delete text;
    }
    void print(int indent)
    {
      Indent(indent);
      printf("Token: %d\tText: \"%s\"\n", type, text->charstar());
    }
};

#endif /* _CToken_H_ */
