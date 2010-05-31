/**
 * ANTLR (v3) Tree Parser for the Charj Language
 */

tree grammar CharjEmitter;

options {
    backtrack = true; 
    memoize = true;
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    output = template;
}


@header {
package charj.translator;
}


@members {
    SymbolTable symtab = null;

    PackageScope currentPackage = null;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
    LocalScope currentLocalScope = null;

    Translator translator_;
    OutputMode mode_;

    private boolean emitCC() { return mode_ == OutputMode.cc; }
    private boolean emitCI() { return mode_ == OutputMode.ci; }
    private boolean emitH() { return mode_ == OutputMode.h; }
    private boolean debug() { return translator_.debug(); }
    private String basename() { return translator_.basename(); }

    /**
     *  Override ANTLR's token mismatch behavior so we throw exceptions early.
     */
    protected void mismatch(IntStream input, int ttype, BitSet follow)
        throws RecognitionException {
        throw new MismatchedTokenException(ttype, input);
    }

    /**
     *  Override ANTLR's set mismatch behavior so we throw exceptions early.
     */
    public Object recoverFromMismatchedSet(IntStream input, RecognitionException e, BitSet follow)
        throws RecognitionException {
        throw e;
    }

    /**
     *  Test a list of CharjAST nodes to see if any of them has the given token
     *  type.
     */
    public boolean listContainsToken(List<CharjAST> list, int tokenType) {
        if (list == null) return false;
        for (CharjAST node : list) {
            if (node.token.getType() == tokenType) {
                return true;
            }
        }
        return false;
    }
}


// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}


// Starting point for parsing a Charj file.
charjSource[SymbolTable symtab, OutputMode m]
@init {
    this.symtab = symtab;
    this.translator_ = symtab.translator;
    this.mode_ = m;
}
    :   ^(CHARJ_SOURCE (p=packageDeclaration)? 
        (i+=importDeclaration)* 
        (t+=typeDeclaration)*)
        -> {emitCC()}? charjSource_cc(basename={basename()}, pd={$p.names}, ids={$i}, tds={$t}, debug={debug()})
        -> {emitCI()}? charjSource_ci(basename={basename()}, pd={$p.names}, ids={$i}, tds={$t}, debug={debug()})
        -> {emitH()}? charjSource_h(basename={basename()}, pd={$p.names}, ids={$i}, tds={$t}, debug={debug()})
        ->
    ;

packageDeclaration
returns [List names]
    :   ^('package' (ids+=IDENT)+)
        {
            $names = $ids;
        }
        ->
    ;
    
importDeclaration
@init {
    String importID = null;
}
    :   ^('import' qualifiedIdentifier ds='.*'?)
        {
            importID = $qualifiedIdentifier.text;
            if ($ds != null) {
            }
        }
        {$ds == null}? // TODO: add support for importing x.*
        -> {(emitCC() || emitH())}? importDeclaration_cc_h(
            inc_id={importID.replace(".","/")},
            use_id={importID.replace(".","::")})
        ->
    ;
    
typeDeclaration
    :   ^(TYPE CLASS IDENT (^('extends' su=type))? (^('implements' type+))?
        {
            currentClass = (ClassSymbol)$IDENT.symbol;
        }
        (csds+=classScopeDeclaration)*)
        -> {emitCC()}? classDeclaration_cc(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitH()}?  classDeclaration_h(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        ->
    |   ^(INTERFACE IDENT (^('extends' type+))? interfaceScopeDeclaration*)
        -> template(t={$text}) "/*INTERFACE-not implemented*/ <t>"
    |   ^(ENUM IDENT (^('implements' type+))? classScopeDeclaration*)
        -> template(t={$text}) "/*ENUM-not implemented*/ <t>"
    |   ^(TYPE chareType IDENT (^('extends' type))? (^('implements' type+))?
        {
            currentClass = (ClassSymbol)$IDENT.symbol;
        }
        (csds+=classScopeDeclaration)*)
        -> {emitCC()}? chareDeclaration_cc(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitCI()}? chareDeclaration_ci(
                sym={currentClass},
                chareType={$chareType.st},
                arrayDim={null},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitH()}? chareDeclaration_h(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        ->
    ;

chareType
@init {
$st = %{$start.getText()};
}
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
        -> template(t={$ARRAY_DIMENSION.text}) "array [<t>]"
    ;

enumConstant
    :   ^(IDENT arguments?)
        -> template(t={$text}) "/*enumConstant-not implemented*/ <t>"
    ;

classScopeDeclaration
@init
{
    boolean entry = false;
}
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
        {
            // determine whether it's an entry method
            if($m.start != null)
                entry = listContainsToken($m.start.getChildren(), CHARJ_MODIFIER_LIST);
        }
        -> {emitCC()}? funcMethodDecl_cc(
                sym={currentClass},
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        -> {emitH()}? funcMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        -> {(emitCI() && entry)}? funcMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        ->
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
        -> {emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
        -> {emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList b=block)
        {
            // determine whether it's an entry method
            if($m.start != null)
                entry = listContainsToken($m.start.getChildren(), CHARJ_MODIFIER_LIST);
        }
        -> {emitCC()}? ctorDecl_cc(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitCI() && entry}? ctorDecl_ci(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitH()}? ctorDecl_h(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        ->
    ;
    
interfaceScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? type IDENT formalParameterList arrayDeclaratorList?)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST (var_decls+=variableDeclarator)+)
        -> var_decl_list(var_decls={$var_decls})
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR id=variableDeclaratorId initializer=variableInitializer?)
        -> var_decl(id={$id.st}, initializer={$initializer.st})
    ;
    
variableDeclaratorId
    :   ^(IDENT adl=arrayDeclaratorList?)
        -> var_decl_id(id={$IDENT.text}, arrayDeclList={$adl.st})
    ;

variableInitializer
    :   arrayInitializer
        -> {$arrayInitializer.st}
    |   expression
        -> {$expression.st}
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
        -> template(t={$text}) "<t>"
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
        -> template(t={$text}) "/* arrayInitializer-not implemented */ <t>"
    ;

genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
        -> template(t={$text}) "/*GENERIC_TYPE_PARAM_LIST-not implemented*/ <t>"
    ;

genericTypeParameter
    :   ^(IDENT bound?)
        -> template(t={$text}) "/*genericTypeParameter-not implemented*/ <t>"
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
        -> template(t={$text}) "/*EXTENDS_BOUND_LIST-not implemented*/ <t>"
    ;

throwsClause
    :   ^(THROWS_CLAUSE qualifiedIdentifier+)
        -> template(t={$text}) "/* throwsClause-not implemented */ <t>"
    ;

modifierList
    :   ^(MODIFIER_LIST accessModifierList? localModifierList? charjModifierList? otherModifierList?)
        ->  {emitCC()}? mod_list_cc(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->  {emitH()}? mod_list_h(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->  {emitCI()}? mod_list_ci(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->
    ;

modifier
    :   accessModifier
    |   localModifier
    |   charjModifier
    |   otherModifier
    ;

accessModifierList
returns [List names]
    :   ^(ACCESS_MODIFIER_LIST (m+=accessModifier)+)
        {
            $names = $m;
        }
    ;
localModifierList
returns [List names]
    :   ^(LOCAL_MODIFIER_LIST (m+=localModifier)+)
        {
            $names = $m;
        }
    ;

charjModifierList
returns [List names]
    :   ^(CHARJ_MODIFIER_LIST (m+=charjModifier)+)
        {
            $names = $m;
        }
    ;

otherModifierList
returns [List names]
    :   ^(OTHER_MODIFIER_LIST (m+=otherModifier)+)
        {
            $names = $m;
        }
    ;
    
localModifier
@init
{
    $st = %{$start.getText()};
}
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

accessModifier
@init
{
    $st = %{$start.getText()};
}
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
    ;

charjModifier
@init
{
    $st = %{$start.getText()};
}
    :   ENTRY
    ;

otherModifier
@init
{
    $st = %{$start.getText()};
}
    :   ABSTRACT
    |   NATIVE
    ;
    
type
    :   simpleType
        -> {$simpleType.st}
    |   objectType 
        -> {$objectType.st}
    |   VOID
        {
            $st = %{$start.getText()};
        }
    ;

simpleType
    :   ^(SIMPLE_TYPE primitiveType arrayDeclaratorList?)
        -> simple_type(typeID={$primitiveType.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

objectType
    :   ^(OBJECT_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> obj_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    |   ^(PROXY_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> proxy_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    |   ^(POINTER_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> pointer_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    |   ^(REFERENCE_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> reference_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT (t+=typeIdent)+) 
        -> template(types={$t}) "<types; separator=\"::\">"
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
        -> typeIdent(typeID={$IDENT.text}, generics={$genericTypeArgumentList.st})
    ;

primitiveType
@init {
$st = %{$start.getText()};
}
    :   BOOLEAN
        -> template() "bool"
    |   CHAR
    |   BYTE
        -> template() "char"
    |   SHORT
    |   INT
    |   LONG
    |   FLOAT
    |   DOUBLE
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST (gta+=genericTypeArgument)+)
        -> template(gtal={$gta}) "\<<gtal; separator=\", \">\>"
    ;
    
genericTypeArgument
    :   type
        -> {$type.st}
    |   '?'
        -> template(t={$text}) "/* genericTypeArgument: wildcard bound types not implemented */ <t>"
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST (fpsd+=formalParameterStandardDecl)* fpvd=formalParameterVarargDecl?)
        -> formal_param_list(sdecl={$fpsd}, vdecl={$fpvd.st})
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL lms=localModifierList? t=type vdid=variableDeclaratorId)
        -> formal_param_decl(modList={$lms.st}, type={$t.st}, declID={$vdid.st})
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId)
        -> template(t={$text}) "/*formal parameter varargs not implemented*/ <t>"
    ;
    
qualifiedIdentifier
    :   IDENT
        -> template(t={$text}) "<t>"
    |   ^(DOT qualifiedIdentifier IDENT)
        -> template(t={$text}) "<t>"
    ;
    
block
@init { boolean emptyBlock = true; }
    :   ^(BLOCK (b+=blockStatement)*)
        { emptyBlock = ($b == null || $b.size() == 0); }
        -> {emitCC() && emptyBlock}? template(bsl={$b}) "{ }"
        -> {emitCC()}? block_cc(bsl={$b})
        ->
    ;
    
blockStatement
    :   localVariableDeclaration
        -> {$localVariableDeclaration.st}
    |   statement
        -> {$statement.st}
    ;


localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList)
        -> local_var_decl(
            modList={null},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList)
        -> local_var_decl(
            modList={null},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
    ;


statement
    :   nonBlockStatement
        -> {$nonBlockStatement.st}
    |   block
        -> {$block.st}
    ;

nonBlockStatement
    :   ^(ASSERT cond=expression msg=expression?)
        -> assert(cond={$cond.st}, msg={$msg.st})
    |   ^(IF parenthesizedExpression then=block else_=block?)
        -> if(cond={$parenthesizedExpression.st}, then={$then.st}, else_={$else_.st})
    |   ^(FOR forInit? FOR_EXPR cond=expression? FOR_UPDATE (update+=expression)* b=block)
        -> for(initializer={$forInit.st}, cond={$cond.st}, update={$update}, body={$block.st})
    |   ^(FOR_EACH localModifierList? type IDENT expression block) 
        -> template(t={$text}) "/* foreach not implemented */ <t>"
    |   ^(WHILE pe=parenthesizedExpression b=block)
        -> while(cond={$pe.st}, body={$b.st})
    |   ^(DO b=block pe=parenthesizedExpression)
        -> dowhile(cond={$pe.st}, block={$b.st})
    |   ^(SWITCH pe=parenthesizedExpression (scls+=switchCaseLabel)*)
        -> switch(expr={$pe.st}, labels={$scls})
    |   ^(RETURN e=expression?)
        -> return(val={$e.st})
    |   ^(THROW expression)
        -> template(t={$text}) "/* throw not implemented */ <t>"
    |   ^(BREAK IDENT?)
        -> template() "break;" // TODO: support labeling
    |   ^(CONTINUE IDENT?)
        -> template() "continue;" // TODO: support labeling
    |   ^(LABELED_STATEMENT i=IDENT s=statement)
        -> label(text={$i.text}, stmt={$s.st})
    |   expression
        -> template(expr={$expression.st}) "<expr>;"
    |   ^('delete' qualifiedIdentifier)
        -> template(t={$qualifiedIdentifier.st}) "delete <t>;"
    |   ^('embed' STRING_LITERAL EMBED_BLOCK)
        ->  embed_cc(str={$STRING_LITERAL.text}, blk={$EMBED_BLOCK.text})
    |   ';' // Empty statement.
        -> {%{$start.getText()}}
    ;
        
switchCaseLabel
    :   ^(CASE expression (b+=blockStatement)*)
        -> case(expr={$expression.st}, block={$b})
    |   ^(DEFAULT (b+=blockStatement)*)
        -> template(block={$b}) "default: <block>"
    ;
    
forInit
    :   localVariableDeclaration
        -> template(lvd={$localVariableDeclaration.st}) "<lvd>"
    |   (ex+=expression)+
        -> template(ex={$ex}) "<ex; separator=\", \">"
    ;

// EXPRESSIONS

parenthesizedExpression
    :   ^(PAREN_EXPR exp=expression)
        -> template(expr={$exp.st}) "(<expr>)"
    ;
    
expression
    :   ^(EXPR expr)
        -> {$expr.st}
    ;

expr
    :   ^(ASSIGNMENT e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> = <e2>"
    |   ^('+=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> += <e2>"
    |   ^('-=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> -= <e2>"
    |   ^('*=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> *= <e2>"
    |   ^('/=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> /= <e2>"
    |   ^('&=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> &= <e2>"
    |   ^('|=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> |= <e2>"
    |   ^('^=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> ^= <e2>"
    |   ^('%=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> %= <e2>"
    |   ^('>>>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>\>= <e2>"
    |   ^('>>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>= <e2>"
    |   ^('<<=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<\<= <e2>"
    |   ^('?' e1=expr e2=expr e3=expr)
        -> template(e1={$e1.st}, e2={$e2.st}, e3={$e3.st}) "<e1> ? <e2> : <e3>"
    |   ^('||' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> || <e2>"
    |   ^('&&' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> && <e2>"
    |   ^(BITWISE_OR e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> | <e2>"
    |   ^('^' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> ^ <e2>"
    |   ^('&' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> & <e2>"
    |   ^(EQUALS e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> == <e2>"
    |   ^('!=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> != <e2>"
    |   ^('instanceof' expr type)
        -> template(t={$text}) "/* instanceof not implemented */ <t>"
    |   ^('<=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<= <e2>"
    |   ^('>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>= <e2>"
    |   ^('>>>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>\> <e2>"
    |   ^('>>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\> <e2>"
    |   ^('>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \> <e2>"
    |   ^('<<' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<\< <e2>"
    |   ^('<' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \< <e2>"
    |   ^('+' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> + <e2>"
    |   ^('-' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> - <e2>"
    |   ^('*' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> * <e2>"
    |   ^('/' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> / <e2>"
    |   ^('%' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> % <e2>"
    |   ^(UNARY_PLUS e1=expr)
        -> template(e1={$e1.st}) "+<e1>"
    |   ^(UNARY_MINUS e1=expr)
        -> template(e1={$e1.st}) "-<e1>"
    |   ^(PRE_INC e1=expr)
        -> template(e1={$e1.st}) "++<e1>"
    |   ^(PRE_DEC e1=expr)
        -> template(e1={$e1.st}) "--<e1>"
    |   ^(POST_INC e1=expr)
        -> template(e1={$e1.st}) "<e1>++"
    |   ^(POST_DEC e1=expr)
        -> template(e1={$e1.st}) "<e1>--"
    |   ^(TILDE e1=expr)
        -> template(e1={$e1.st}) "~<e1>"
    |   ^(NOT e1=expr)
        -> template(e1={$e1.st}) "!<e1>"
    |   ^(CAST_EXPR ty=type e1=expr)
        -> template(ty={$ty.st}, e1={$e1.st}) "(<ty>)<e1>"
    |   primaryExpression
        -> {$primaryExpression.st}
    ;

primaryExpression
    :   ^(DOT prim=primaryExpression
            ( IDENT   -> template(id={$IDENT}, prim={$prim.st}) "<prim>.<id>"
            | THIS    -> template(prim={$prim.st}) "<prim>.this"
            | SUPER   -> template(prim={$prim.st}) "<prim>.super"
            )
        )
    |   ^(ARROW prim=primaryExpression
            ( IDENT   -> template(id={$IDENT}, prim={$prim.st}) "<prim>-><id>"
            | THIS    -> template(prim={$prim.st}) "<prim>->this"
            | SUPER   -> template(prim={$prim.st}) "<prim>->super"
            )
        )
    |   parenthesizedExpression
        -> {$parenthesizedExpression.st}
    |   IDENT
        -> {%{$start.getText()}}
    |   ^(METHOD_CALL pe=primaryExpression gtal=genericTypeArgumentList? args=arguments)
        -> method_call(primary={$pe.st}, generic_types={$gtal.st}, args={$args.st})
    |   ^(ENTRY_METHOD_CALL pe=primaryExpression gtal=genericTypeArgumentList? args=arguments)
        -> method_call(primary={$pe.st}, generic_types={$gtal.st}, args={$args.st})
    |   explicitConstructorCall
        -> {$explicitConstructorCall.st}
    |   ^(ARRAY_ELEMENT_ACCESS pe=primaryExpression ex=expression)
        -> template(pe={$pe.st}, ex={$ex.st}) "<pe>[<ex>]"
    |   literal
        -> {$literal.st}
    |   newExpression
        -> {$newExpression.st}
    |   THIS
        -> {%{$start.getText()}}
    |   arrayTypeDeclarator
        -> {$arrayTypeDeclarator.st}
    |   SUPER
        -> {%{$start.getText()}}
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
        -> template(t={$text}) "<t>"
    ;

newExpression
    :   ^(  STATIC_ARRAY_CREATOR
            (   primitiveType newArrayConstruction
            |   genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction
            )
        )
        -> template(t={$text}) "<t>"
    |   ^(NEW qualifiedTypeIdent arguments)
        -> template(q={$qualifiedTypeIdent.st}, a={$arguments.st}) "new <q>(<a>)"
    ;

newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
        -> array_construction_with_init(
                array_decls={$arrayDeclaratorList.st},
                initializer={$arrayInitializer.st})
    |   (ex+=expression)+ adl=arrayDeclaratorList?
        -> array_construction(exprs={$ex}, array_decls={$adl.st})
    ;

arguments
    :   ^(ARGUMENT_LIST (ex+=expression)*)
        -> arguments(exprs={$ex})
    ;

literal
@init {
$st = %{$start.getText()};
}
    :   HEX_LITERAL
    |   OCTAL_LITERAL
    |   DECIMAL_LITERAL
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   TRUE
    |   FALSE
    |   NULL
    ;

