
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
# define NODEGROUP 263
# define ARRAY 264
# define MESSAGE 265
# define CLASS 266
# define STACKSIZE 267
# define THREADED 268
# define TEMPLATE 269
# define SYNC 270
# define VOID 271
# define PACKED 272
# define VARSIZE 273
# define ENTRY 274
# define MAINCHARE 275
# define IDENT 276
# define NUMBER 277
# define LITERAL 278
# define INT 279
# define LONG 280
# define SHORT 281
# define CHAR 282
# define FLOAT 283
# define DOUBLE 284
# define UNSIGNED 285
