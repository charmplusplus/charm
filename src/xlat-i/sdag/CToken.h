#ifndef _CToken_H_
#define _CToken_H_

#include "EToken.h"
#include "sdag-globals.h"
#include "xi-util.h"
#include <stdio.h>

namespace xi {

class CToken {
 public:
  EToken type;
  XStr* text;
  CToken(EToken t, const char* txt) {
    type = t;
    text = new XStr(txt);
  }
  ~CToken() {
    // delete text;
  }
  void print(int indent) {
    Indent(indent);
    printf("Token: %d\tText: \"%s\"\n", type, text->charstar());
  }
};

}  // namespace xi

#endif /* _CToken_H_ */
