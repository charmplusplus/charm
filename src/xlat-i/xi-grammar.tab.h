
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
# define ARRAY 263
# define MESSAGE 264
# define CLASS 265
# define STACKSIZE 266
# define THREADED 267
# define TEMPLATE 268
# define SYNC 269
# define VOID 270
# define PACKED 271
# define VARSIZE 272
# define ENTRY 273
# define MAINCHARE 274
# define IDENT 275
# define NUMBER 276
# define LITERAL 277
# define INT 278
# define LONG 279
# define SHORT 280
# define CHAR 281
# define FLOAT 282
# define DOUBLE 283
# define UNSIGNED 284
