%{
#include <stdio.h>
#include <string.h>
int tok_lineno = 0;
int yytoktype;

/******************************************************************************
 *
 * Files
 *
 *****************************************************************************/

FILE *file_src;
FILE *file_cpm;

/******************************************************************************
 *
 * Lexical Analyzer
 *
 *****************************************************************************/
%}

%e 1500
%p 3900

STR    ([^"\\\n]|\\(['"?\\abfnrtv\n]|[0-7]{1,3}|[xX][0-9a-fA-F]{1,3}))
EXP    ([eE][+-]?[0-9]+)
FS     [flFL]
IS     ([uU][lL]?|[lL][uU]?)
ID     [_a-zA-Z][_a-zA-Z0-9]*

%%

"CpmDeclarePointer1"        { return 'P'; }
"CpmDeclareSimple1"         { return 'S'; }
"CpmInvokable"              { return 'I'; }
{ID}                        { return 'i'; }
[0-9]+"."[0-9]*{EXP}?{FS}?  { return '0'; }
"."[0-9]+{EXP}?{FS}?        { return '0'; }
[0-9]+{EXP}{FS}?            { return '0'; }
[1-9][0-9]*{IS}?            { return '0'; }
0[0-7]*{IS}?                { return '0'; }
0[xX][0-9a-fA-F]+{IS}?      { return '0'; }
\"{STR}*\"          	    { return '"'; }
\'{STR}+\"                  { return 'c'; }
[ \t\f]+                    { }
[\n]                        { tok_lineno++; }
.   		            { return yytext[0]; }

%%


#ifdef yywrap
#undef yywrap
#endif
yywrap(){ return(1); }

void yytokget()
{
  yytoktype=yylex();
}

void yychkword(char *s)
{
  if (strcmp(yytext, s)) {
    fprintf(stderr,"%s expected\n", s);
    exit(1);
  }
}

void yychktype(char c)
{
  if (yytoktype == c) return;
  if (c=='i') {
    fprintf(stderr,"identifier expected\n");
    exit(1);
  } else {
    fprintf(stderr,"%c expected\n",c);
    exit(1);
  }
}

/******************************************************************************
 *
 * The 'symbol table', such as it is.
 *
 *****************************************************************************/

char *mod_funcs[1000];
int   mod_len;

char *type_name[1000];
int   type_kind[1000];
int   type_count;

char *func_name;
char *func_args[1000];
int   func_array[1000];
int   func_pointer[1000];
int   func_len;
int   func_static;

int type_simple(char *type)
{
  int i;
  for (i=0; i<type_count; i++) {
    if (strcmp(type, type_name[i])==0)
      return (type_kind[i]=='S') ? 1:0;
  }
  fprintf(stderr,"Unknown type %s\n", type);
  exit(1); return 0;
}

void type_declare(char *type, int kind)
{
  if (type_count==1000) {
    fprintf(stderr,"error: type table overflow.\n");
    exit(1);
  }
  type_name[type_count] = strdup(type);
  type_kind[type_count] = kind;
  type_count++;
}

/******************************************************************************
 *
 * Code Generation
 *
 *****************************************************************************/

void gen_actual_args(FILE *f)
{
  int i;
  fprintf(f, "(");
  if (func_len) {
    fprintf(f, "CpmA0");
    for (i=1; i<func_len; i++) {
      fprintf(f, ",CpmA%d", i);
    }
  }
  fprintf(f,");\n");
}

void gen_dimension_required()
{
  fprintf(stderr,"CpmDim required before array.\n");
  exit(1);
}

void gen_func_struct()
{
  int i;
  fprintf(file_cpm, "struct CpmSt_%s\n", func_name);
  fprintf(file_cpm, "{\n");
  fprintf(file_cpm, "char convcore[CmiMsgHeaderSizeBytes];\n");
  fprintf(file_cpm, "unsigned int envpos;\n");
  for (i=0; i<func_len; i++) {
    if ((func_pointer[i]==0) && (func_array[i]==0)) {
      fprintf(file_cpm, "%s f%d;\n",func_args[i],i);
    } else {
      fprintf(file_cpm, "int f%d;\n",i);
    }
  }
  fprintf(file_cpm, "};\n");
}

void gen_func_recv()
{
  int i;
  fprintf(file_cpm, "static void CpmRecv_%s(char *CpmM)\n", func_name);
  fprintf(file_cpm, "{\n");
  fprintf(file_cpm, "struct CpmSt_%s *CpmS=(struct CpmSt_%s *)CpmM;\n",
	  func_name, func_name);
  fprintf(file_cpm, "char *CpmX = (char *)CpmS;\n");
  fprintf(file_cpm, "int i;\n");
  for (i=0; i<func_len; i++) {
    fprintf(file_cpm, "%s %sCpmA%d;\n", func_args[i], func_array[i]?"*":"", i);
  }
  for (i=0; i<func_len; i++) {
    int mode = (func_pointer[i] ? 2 : 0) + (func_array[i] ? 1 : 0);
    switch(mode) {
    case 0: /* simple */
      fprintf(file_cpm, "CpmA%d = CpmS->f%d;\n", i, i);
      fprintf(file_cpm, "CpmUnpack_%s(CpmA%d);\n", func_args[i], i);
      break;
    case 1: /* array of simple */
      if ((i==0)||(func_array[i-1])||(strcmp(func_args[i-1],"CpmDim")))
	gen_dimension_required();
      fprintf(file_cpm, "CpmA%d = (%s *)(CpmX+(CpmS->f%d));\n", i, func_args[i], i);
      fprintf(file_cpm, "for (i=0; i<CpmA%d; i++)\n", i-1);
      fprintf(file_cpm, "  CpmUnpack_%s(CpmA%d[i]);\n", func_args[i], i);
      break;
    case 2: /* pointer */
      fprintf(file_cpm, "CpmA%d = (%s)(CpmX+(CpmS->f%d));\n", i, func_args[i], i);
      fprintf(file_cpm, "CpmPtrUnpack_%s(CpmA%d);\n", func_args[i], i);
      break;
    case 3: /* array of pointer */
      if ((i==0)||(func_array[i-1])||(strcmp(func_args[i-1],"CpmDim")))
	gen_dimension_required();
      fprintf(file_cpm, "CpmA%d = (%s *)(CpmX+(CpmS->f%d));\n", i, func_args[i], i);
      fprintf(file_cpm, "for (i=0; i<CpmA%d; i++) {\n", i-1);
      fprintf(file_cpm, "  CpmA%d[i] = CpmM + (size_t)(CpmA%d[i]);\n", i, i);
      fprintf(file_cpm, "  CpmPtrUnpack_%s(CpmA%d[i]);\n", func_args[i], i);
      fprintf(file_cpm, "}\n");
      break;
    }
  }
  fprintf(file_cpm,"%s", func_name);
  gen_actual_args(file_cpm);
  fprintf(file_cpm, "}\n");
}

void gen_func_send()
{
  int i;
  if (func_static) fprintf(file_cpm, "static ");

  fprintf(file_cpm,"void *Cpm_%s(CpmDestination ctrl",func_name);
  for (i=0; i<func_len; i++) {
    fprintf(file_cpm, ",%s %sa%d", func_args[i], func_array[i]?"*":"", i);
  }
  fprintf(file_cpm, ")\n");
  fprintf(file_cpm, "{\n");
  fprintf(file_cpm, "struct CpmSt_%s *msg;\n",func_name);
  fprintf(file_cpm, "char *data; int size, i, envpos; void *result;\n");
  fprintf(file_cpm, "int offs = CpmAlign(sizeof(struct CpmSt_%s), double);\n",func_name);
  for (i=0; i<func_len; i++) {
    if (func_array[i])
      fprintf(file_cpm, "int aoffs%d;\n",i);
    if (func_pointer[i]) {
      if (func_array[i]) {
	fprintf(file_cpm, "size_t *poffs%d = (size_t *)malloc(a%d*sizeof(size_t));\n",i,i-1);
      } else {
	fprintf(file_cpm, "int poffs%d;\n",i);
      }
    }
  }
  fprintf(file_cpm, "envpos=offs; offs=CpmAlign(offs+(ctrl->envsize),double);\n");
  for (i=0; i<func_len; i++) {
    if (func_array[i]) {
      fprintf(file_cpm, "size=a%d*sizeof(%s);\n",i-1,func_args[i]);
      fprintf(file_cpm, "aoffs%d=offs; offs=CpmAlign(offs+size,double);\n",i);
    }
  }
  for (i=0; i<func_len; i++) {
    if (func_pointer[i]) {
      if (func_array[i]) {
	fprintf(file_cpm, "for (i=0; i<a%d; i++) {\n",i-1) ;
	fprintf(file_cpm, "  size = CpmPtrSize_%s(a%d[i]);\n",func_args[i],i);
	fprintf(file_cpm, "  poffs%d[i]=offs; offs=CpmAlign(offs+size,double);\n",i);
	fprintf(file_cpm, "}\n");
      } else {
	fprintf(file_cpm, "size = CpmPtrSize_%s(a%d);\n",func_args[i],i);
	fprintf(file_cpm, "poffs%d=offs; offs=CpmAlign(offs+size, double);\n",i);
      }
    }
  }
  fprintf(file_cpm, "data = (char *)CmiAlloc(offs);\n");
  fprintf(file_cpm, "msg = (struct CpmSt_%s *)data;\n",func_name);
  fprintf(file_cpm, "msg->envpos = envpos;\n");
  for (i=0; i<func_len; i++) {
    int mode = (func_array[i]?2:0) | (func_pointer[i]?1:0);
    switch(mode) {
    case 0: /* one simple */
      fprintf(file_cpm, "CpmPack_%s(a%d);\n",func_args[i],i);
      fprintf(file_cpm, "msg->f%d = a%d;\n",i,i);
      break;
    case 1: /* one pointer */
      fprintf(file_cpm, "msg->f%d = poffs%d;\n",i,i);
      fprintf(file_cpm, "CpmPtrPack_%s(((%s)(data+poffs%d)), a%d);\n",
	      func_args[i],func_args[i],i,i);
      break;
    case 2: /* array simple */
      fprintf(file_cpm, "msg->f%d = aoffs%d;\n",i,i);
      fprintf(file_cpm, "memcpy(data+aoffs%d, a%d, a%d*sizeof(%s));\n",
	      i,i,i-1,func_args[i]);
      fprintf(file_cpm, "for(i=0; i<a%d; i++)\n",i-1);
      fprintf(file_cpm, "  CpmPack_%s(((%s *)(data+aoffs%d))[i]);\n",
	      func_args[i],func_args[i],i);
      break;
    case 3: /* array pointer */
      fprintf(file_cpm, "msg->f%d = aoffs%d;\n",i,i);
      fprintf(file_cpm, "memcpy(data+aoffs%d, poffs%d, a%d*sizeof(size_t));\n",
	      i,i,i-1);
      fprintf(file_cpm, "for(i=0; i<a%d; i++)\n",i-1);
      fprintf(file_cpm, "  CpmPtrPack_%s(((%s)(data+(poffs%d[i]))), a%d[i]);\n",
	      func_args[i],func_args[i], i,i);
      break;
    }
  }
  fprintf(file_cpm,"CmiSetHandler(msg, CpvAccess(CpmIndex_%s));\n",func_name);
  fprintf(file_cpm,"result = (ctrl->sendfn)(ctrl, offs, msg);\n");
  for (i=0; i<func_len; i++)
    if ((func_pointer[i])&&(func_array[i]))
      fprintf(file_cpm, "free(poffs%d);\n", i);
  fprintf(file_cpm,"return result;\n");
  fprintf(file_cpm,"}\n");
}

void gen_func_protos()
{
  int i;

  fprintf(file_cpm, "CpvStaticDeclare(int, CpmIndex_%s);\n", func_name);

  fprintf(file_cpm,"void %s(",func_name);
  if (func_len) {
    fprintf(file_cpm, "%s %s", func_args[0], func_array[0]?"*":"");
    for (i=1; i<func_len; i++)
      fprintf(file_cpm, ",%s %s", func_args[i], func_array[i]?"*":"");
  }
  fprintf(file_cpm,");\n");
}

void gen_func_c()
{
  gen_func_protos();
  gen_func_struct();
  gen_func_recv();
  gen_func_send();
}

void gen_mod_head()
{
  fprintf(file_cpm, "CpvStaticDeclare(int, CpmIPRIO);\n");
}

void gen_mod_tail()
{
  int i;
  fprintf(file_cpm, "static void CpmInitializeThisModule()\n");
  fprintf(file_cpm, "{\n");
  fprintf(file_cpm, "CpvInitialize(int, CpmIPRIO);\n");
  for (i=0; i<mod_len; i++) {
    fprintf(file_cpm, "CpvInitialize(int, CpmIndex_%s);\n", mod_funcs[i]);
    fprintf(file_cpm, "CpvAccess(CpmIndex_%s) = CmiRegisterHandler(CpmRecv_%s);\n",
	  mod_funcs[i], mod_funcs[i]);
  }
  fprintf(file_cpm, "}\n");
}

/******************************************************************************
 *
 * The Parser
 *
 *****************************************************************************/

void parse_type()
{
  int kind = yytoktype;
  yytokget();
  if (strncmp(yytext,"CpmType_",8)==0) {
    type_declare(yytext+8, kind);
  }
  yytokget();
}

void parse_list()
{
  int level=1;
  yychktype('{');
  yytokget();
  while (level) {
    if (yytoktype == '{') level++;
    if (yytoktype == '}') level--;
    if (yytoktype==0) return;
    yytokget();
  }
}

void parse_func()
{
  yychkword("CpmInvokable");
  yytokget();
  if (yytoktype==';')
    return;
  func_len=0;
  func_static=0;
  if (strcmp(yytext,"static")==0) {
    func_static=1;
    yytokget();
  }
  yychktype('i');
  func_name = strdup(yytext);
  yytokget();
  yychktype('(');
  yytokget();
  if (yytoktype != ')') {
    while (1) {
      yychktype('i');
      func_args[func_len] = strdup(yytext);
      func_array[func_len] = 0;
      yytokget();
      if (yytoktype == '*') {
	yytokget();
	func_array[func_len] = 1;
      }
      func_pointer[func_len] = !type_simple(func_args[func_len]);
      func_len++;
      yychktype('i');
      yytokget();
      if (yytoktype == ')') break;
      yychktype(',');
      yytokget();
    }
  }
  yytokget();
  mod_funcs[mod_len++] = func_name;
  gen_func_c();
}

void parse_all()
{  
  yyin = file_src;
  yytokget();
  gen_mod_head();
  top:
  switch (yytoktype) {
  case 0: break;
  case 'P': parse_type(); goto top;
  case 'S': parse_type(); goto top;
  case 'I': parse_func(); goto top;
  case '{': parse_list(); goto top;
  default: yytokget(); goto top; 
  }
  gen_mod_tail();
}

/******************************************************************************
 *
 * Setup
 *
 *****************************************************************************/

void usage()
{
  fprintf(stderr,"usage: cpm <modulename>\n");
  exit(1);
}

FILE *fopen_nofail(char *path, char *mode)
{
  FILE *res = fopen(path, mode);
  if (res==0) {
    fprintf(stderr,"Couldn't open %s with mode %s\n",path,mode);
    exit(1);
  }
  return res;
}

void disclaim(FILE *f, char *src)
{
  fprintf(f,"/* WARNING: This file generated by Converse */\n");
  fprintf(f,"/* Marshalling System from source-file %s. */\n",src);
  fprintf(f,"\n");
}

main(int argc, char **argv)
{
  if (argc != 3) usage();
  file_src = fopen_nofail(argv[1], "r");
  file_cpm = fopen_nofail(argv[2], "w");
  disclaim(file_cpm, argv[1]);
  parse_all();
  exit(0);
}
