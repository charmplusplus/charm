#include "ckregex.h"
#include <regex>
#include <cstring>

extern "C" char ** findFirstCaptures(const char * pattern, const char * input_str) {
  std::regex re(pattern);
  std::smatch match;
  std::string s(input_str);
  if (std::regex_search(s, match, re)) {
    const size_t match_size = match.size();
    // allocate size number of slots for result because size-1 matches and 1 for terminating NULL
    char ** result = (char **)malloc(match_size * sizeof(char *));
    result[match_size - 1] = NULL;

    // copy over all capture, start from index 1 since index 0 contains the full match not a capture
    for (int i = 1; i < match_size; ++i) {
      result[i-1] = strdup(match.str(i).c_str());
    }

    return result;
  } else {
    return NULL;
  }
}
