#include "xi-util.h"
#include "xi-Template.h"

namespace xi {

void 
XStr::append(const char *_s) 
{
  len += strlen(_s);
  if ( len >= blklen) {
    while ( len >= blklen ) {
      blklen += SZ;
    }
    char *tmp = s;
    s = new char[blklen];
    strcpy(s, tmp);
    delete[] tmp;
  }
  strcat(s, _s);
}

void 
XStr::append(char c) 
{
  char tmp[2];
  tmp[0] = c;
  tmp[1] = '\0';
  append(tmp);
}

void XStr::initTo(const char *_s)
{
  len = strlen(_s);
  blklen = SZ;
  while ( len >= blklen ) {
    blklen += SZ;
  }
  s = new char[blklen];
  strcpy(s, _s);
}

XStr::XStr() {initTo("");}
XStr::XStr(const char *_s) {initTo(_s);}
XStr::XStr(const XStr &_s) {initTo(_s.get_string_const());}

void XStr::clear() {
  delete[] s;
  initTo("");
}

XStr& XStr::operator << (int i) {
      char tmp[100]; 
      sprintf(tmp, "%d", i); 
      append(tmp); 
      return *this;
}

void XStr::line_append(const char c)
{
  XStr xs;
  for(unsigned int i=0; i<len; i++) {
    if(s[i] == '\n')
      xs << c << "\n";
    else
      xs << s[i];
  }
  delete[] s;
  initTo(xs.charstar());
}

void XStr::line_append_padding(const char c, int lineWidth)
{
  XStr xs;
  int count = 0;

  for(unsigned int i=0; i<len; i++) {
    if(s[i] == '\n'){
      // found line ending
      while(count++ < lineWidth-1)
	xs << " ";
      xs << c << "\n";
      count=0;
    } else if(s[i] == '\t') {
      // found tab, convert to 2 spaces
      xs << "  ";
      count+=2;
    } else {
      // found non-line ending
      xs << s[i];
      count++;
    }
  }
  delete[] s;
  initTo(xs.charstar());
}



void 
XStr::spew(const char*b, const char *a1, const char *a2, const char *a3, 
           const char *a4, const char *a5)
{
  using std::cout;
  int i,length=strlen(b);
  for(i=0; i<length; i++){
    switch(b[i]){
    case '\001':
      if(a1==0) {cout << "Internal Error\n"; abort();} append(a1); break;
    case '\002':
      if(a2==0) {cout << "Internal Error\n"; abort();} append(a2); break;
    case '\003':
      if(a3==0) {cout << "Internal Error\n"; abort();} append(a3); break;
    case '\004':
      if(a4==0) {cout << "Internal Error\n"; abort();} append(a4); break;
    case '\005':
      if(a5==0) {cout << "Internal Error\n"; abort();} append(a5); break;
    default:
      append(b[i]);
    }
  }
}

void XStr::replace (const char a, const char b) {
  for(unsigned int i=0; i<len; i++) {
    if (s[i] == a) s[i] = b;
  }
}

extern const char *cur_file;

// Fatal error function
void die(const char *why, int line)
{
	if (line==-1)
		fprintf(stderr,"%s: Charmxi fatal error> %s\n",cur_file,why);
	else
		fprintf(stderr,"%s:%d: Charmxi fatal error> %s\n",cur_file,line,why);
	exit(1);
}

char* fortranify(const char *s, const char *suff1, const char *suff2, const char *suff3)
{
  int i, len1 = strlen(s), len2 = strlen(suff1),
         len3 = strlen(suff2), len4 = strlen(suff3);
  int c = len1+len2+len3+len4;
  char str[1024], strUpper[1024];
  strcpy(str, s);
  strcat(str, suff1);
  strcat(str, suff2);
  strcat(str, suff3);
  for (i = 0; i < c+1; i++)
    str[i] = tolower(str[i]);
  for (i = 0; i < c+1; i++)
    strUpper[i] = toupper(str[i]);
  char *retVal;
  retVal = new char[2*c+20];
  strcpy(retVal, "FTN_NAME(");
  strcat(retVal, strUpper);
  strcat(retVal, ",");
  strcat(retVal, str);
  strcat(retVal, ")");

  return retVal;
}

XStr generateTemplateSpec(TVarList* tspec, bool printDefault)
{
  XStr str;

  if(tspec) {
    str << "template < ";
    tspec->genLong(str, printDefault);
    str << " > ";
  }

  return str;
}

const char *forWhomStr(forWhom w)
{
  switch(w) {
  case forAll: return Prefix::Proxy;
  case forIndividual: return Prefix::ProxyElement;
  case forSection: return Prefix::ProxySection;
  case forIndex: return Prefix::Index;
  case forPython: return "";
  default: return NULL;
  };
}

// Make the name lower case
void templateGuardBegin(bool templateOnly, XStr &str) {
  if (templateOnly)
    str << "#ifdef " << "CK_TEMPLATES_ONLY\n";
  else
    str << "#ifndef " << "CK_TEMPLATES_ONLY\n";
}
void templateGuardEnd(XStr &str) {
  str << "#endif /* CK_TEMPLATES_ONLY */\n";
}

// This replaces the line containing a single '#' token with a '#line'
// directive, as described in AtomicConstruct::generateCode.
std::string addLineNumbers(char *str, const char *filename)
{
  int lineNo = 1;
  std::string s(str);
  for (int i = 0; i < s.length(); ++i) {
    switch (s[i]) {
      case '\n':
        lineNo++;
        break;
      case '#':
        if (i > 0 && s[i-1] == '\n' && s[i+1] == '\n') {
          std::stringstream ss;
          ss << "#line " << lineNo+1 << " \"" << filename << "\"";
          s.replace(s.begin() + i, s.begin() + i + 1, ss.str());
        }
    }
  }
  return s;
}

// The following three functions are a workaround for bug #734.
void sanitizeRange(std::string &code, int i, int j)
{
  for (int k = i; k <= j; ++k) {
    switch (code[k]) {
      case '{': code[k] = 0x0E; break;
      case '}': code[k] = 0x0F; break;
      case '(': code[k] = 0x10; break;
      case ')': code[k] = 0x11; break;
      case '[': code[k] = 0x12; break;
      case ']': code[k] = 0x13; break;
      case ';': code[k] = 0x14; break;
      case ':': code[k] = 0x15; break;
      case ',': code[k] = 0x16; break;
      default: break;
    }
  }
}

void desanitizeCode(std::string &code)
{
  for (int i = 0; i < code.size(); ++i) {
    switch (code[i]) {
      case 0x0E: code[i] = '{'; break;
      case 0x0F: code[i] = '}'; break;
      case 0x10: code[i] = '('; break;
      case 0x11: code[i] = ')'; break;
      case 0x12: code[i] = '['; break;
      case 0x13: code[i] = ']'; break;
      case 0x14: code[i] = ';'; break;
      case 0x15: code[i] = ':'; break;
      case 0x16: code[i] = ','; break;
      default: break;
    }
  }
}

void sanitizeComments(std::string &code)
{
  int h, i;
  for (i = 0; i < code.size() - 1; ++i) {
    if (code[i] == '/') {
      h = i+2;
      switch (code[i+1]) {
        case '*':
          // Case 1: /* */
          i += 2;
          for (; !(code[i] == '*' && code[i+1] == '/'); ++i);
          break;

        case '/':
          // Case 2: //
          while (code[++i] != '\n');
          break;

        default:
          // Case 3: not a comment
          continue;
          break;
      }

      sanitizeRange(code, h, i-1);
      ++i;
    }
  }
}

void sanitizeStrings(std::string &code)
{
  int h, i;
  bool in_string = false;
  for (i = 0; i < code.size(); ++i) {
    if (code[i] == '\\') {
      // The next character cannot possibly end a string since this '\'
      // would escape it; just skip over it.
      ++i;
    } else if (code[i] == '"') {
      if (in_string) {
        sanitizeRange(code, h, i-1);
        in_string = false;
      } else {
        h = i+1;
        in_string = true;
      }
    }
  }
}

// charmxi error printing methods
std::string _get_caret_line(int err_line_start, int first_col, int last_col)
{
  std::string caret_line(first_col - err_line_start - 1, ' ');
  caret_line += std::string(last_col - first_col + 1, '^');

  return caret_line;
}

void _pretty_header(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  std::cerr << cur_file << ":" << first_line << ":";

  if (first_col != -1)
    std::cerr << first_col << "-" << last_col << ": ";

  std::cerr << type << ": " << msg << std::endl;
}

void _pretty_print(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  _pretty_header(type, msg, first_col, last_col, first_line, last_line);

  if (first_line <= inputBuffer.size() &&
      first_line <= last_line &&
      first_col <= last_col) {
    std::string err_line = inputBuffer[first_line-1];

    if (err_line.length() != 0) {
      int err_line_start = err_line.find_first_not_of(" \t\r\n");
      err_line.erase(0, err_line_start);

      std::string caret_line;
      if (first_col != -1)
        caret_line = _get_caret_line(err_line_start, first_col, last_col);

      std::cerr << "  " << err_line << std::endl;

      if (first_col != -1)
        std::cerr << "  " << caret_line;
      std::cerr << std::endl;
    }
  }
}

void pretty_msg(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  if (first_line == -1) first_line = lineno;
  if (last_line  == -1)  last_line = lineno;
  _pretty_print(type, msg, first_col, last_col, first_line, last_line);
}

void pretty_msg_noline(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  if (first_line == -1) first_line = lineno;
  if (last_line  == -1)  last_line = lineno;
  _pretty_header(type, msg, first_col, last_col, first_line, last_line);
}

}   // namespace xi

namespace Prefix {

const char *Proxy = "CProxy_";
const char *ProxyElement = "CProxyElement_";
const char *ProxySection = "CProxySection_";
const char *Message = "CMessage_";
const char *Index = "CkIndex_";
const char *Python = "CkPython_";

}   // namespace Prefix
