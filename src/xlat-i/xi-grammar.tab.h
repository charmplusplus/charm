typedef union {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Parameter *pname;
  ParamList *plist;
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
  Chare::attrib_t cattr;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	CHARE	261
#define	MAINCHARE	262
#define	GROUP	263
#define	NODEGROUP	264
#define	ARRAY	265
#define	MESSAGE	266
#define	CLASS	267
#define	STACKSIZE	268
#define	THREADED	269
#define	TEMPLATE	270
#define	SYNC	271
#define	EXCLUSIVE	272
#define	VIRTUAL	273
#define	MIGRATABLE	274
#define	CREATEHERE	275
#define	CREATEHOME	276
#define	VOID	277
#define	CONST	278
#define	PACKED	279
#define	VARSIZE	280
#define	ENTRY	281
#define	IDENT	282
#define	NUMBER	283
#define	LITERAL	284
#define	CPROGRAM	285
#define	INT	286
#define	LONG	287
#define	SHORT	288
#define	CHAR	289
#define	FLOAT	290
#define	DOUBLE	291
#define	UNSIGNED	292


extern YYSTYPE yylval;
