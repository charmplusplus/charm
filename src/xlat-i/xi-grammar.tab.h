
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  EnType *rtype;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  char *strval;
  int intval;
} YYSTYPE;
extern YYSTYPE yylval;
# define MODULE 257
# define MAINMODULE 258
# define EXTERN 259
# define READONLY 260
# define CHARE 261
# define GROUP 262
# define MESSAGE 263
# define CLASS 264
# define STACKSIZE 265
# define THREADED 266
# define TEMPLATE 267
# define SYNC 268
# define VOID 269
# define PACKED 270
# define VARSIZE 271
# define ENTRY 272
# define MAINCHARE 273
# define IDENT 274
# define NUMBER 275
# define LITERAL 276
# define INT 277
# define LONG 278
# define SHORT 279
# define CHAR 280
# define FLOAT 281
# define DOUBLE 282
# define UNSIGNED 283
