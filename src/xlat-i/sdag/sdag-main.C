/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "CLexer.h"
#include "CToken.h"
#include "CParsedFile.h"
#include "CParser.h"
#include "sdag-globals.h"
#include "xi-util.h"

void sdag_trans(XStr& classname, XStr& input, XStr& output)
{
  resetNumbers();
  CParser *cParser = new CParser(input);
  CParsedFile *parsedFile = cParser->doParse();
  parsedFile->doProcess(classname, output);
  delete parsedFile;
  delete cParser;
}
