typedef union {
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
  MsgVar *mv;
  MsgVarList *mvlist;
  char *strval;
  int intval;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	CHARE	261
#define	GROUP	262
#define	NODEGROUP	263
#define	ARRAY	264
#define	MESSAGE	265
#define	CLASS	266
#define	STACKSIZE	267
#define	THREADED	268
#define	MIGRATABLE	269
#define	TEMPLATE	270
#define	SYNC	271
#define	EXCLUSIVE	272
#define	VIRTUAL	273
#define	VOID	274
#define	PACKED	275
#define	VARSIZE	276
#define	VARRAYS	277
#define	ENTRY	278
#define	MAINCHARE	279
#define	IDENT	280
#define	NUMBER	281
#define	LITERAL	282
#define	INT	283
#define	LONG	284
#define	SHORT	285
#define	CHAR	286
#define	FLOAT	287
#define	DOUBLE	288
#define	UNSIGNED	289


extern YYSTYPE yylval;
