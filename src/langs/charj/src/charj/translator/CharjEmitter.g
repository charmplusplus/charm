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
    SymbolTable symtab_ = null;

    PackageScope currentPackage_ = null;
    ClassSymbol currentClass_ = null;
    MethodSymbol currentMethod_ = null;
    LocalScope currentLocalScope_ = null;

    Translator translator_;
    OutputMode mode_;

    private boolean emitCC() { return mode_ == OutputMode.cc; }
    private boolean emitCI() { return mode_ == OutputMode.ci; }
    private boolean emitH() { return mode_ == OutputMode.h; }
    private boolean debug() { return translator_.debug(); }

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
    this.symtab_ = symtab;
    this.translator_ = symtab.translator;
    this.mode_ = m;
    String closingBraces = "";
}
    :   ^(CHARJ_SOURCE (p=packageDeclaration)? 
        (i+=importDeclaration)* 
        (t=typeDeclaration))
        {
            // construct string of }'s to close namespace 
            if ($p.st != null) {
                String temp_p = $p.st.toString();
                for (int idx=0; idx<temp_p.length(); ++idx) {
                    if (temp_p.charAt(idx) == '{') {
                        closingBraces += "} ";
                    }
                }
            }
        }
        -> {emitCC()}? charjSource_cc(
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces}, debug={debug()})
        -> {emitCI()}? charjSource_ci(pd={$p.st}, ids={$i}, tds={$t.st}, debug={debug()})
        -> {emitH()}? charjSource_h(
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces}, debug={debug()})
        ->
    ;

packageDeclaration
@init { 
    List<String> names = null; 
}
    :   ^('package' qualifiedIdentifier)  {
            names =  java.util.Arrays.asList(
                    $qualifiedIdentifier.text.split("[.]"));
        }
        -> {(emitCC() || emitH())}? packageDeclaration_cc_h(
            ids={names})
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
    :   ^('template' '<' 'class' IDENT '>' td=typeDeclaration)
        -> {emitH()}? templateDeclaration_h(
            ident={$IDENT.text},
            class1={$td.st})
        ->
    |   ^('class' i1=IDENT (^('extends' su=type))? (^('implements' type+))? (csds+=classScopeDeclaration)*)
        -> {emitCC()}? classDeclaration_cc(
                ident={$i1.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitH()}? classDeclaration_h(
                ident={$i1.text}, 
                ext={$su.st}, 
                csds={$csds})
        ->
    |   ^('interface' IDENT (^('extends' type+))? interfaceScopeDeclaration*)
        -> template(t={$text}) "/*INTERFACE-not implemented*/ <t>"
    |   ^('enum' IDENT (^('implements' type+))? classScopeDeclaration*)
        -> template(t={$text}) "/*ENUM-not implemented*/ <t>"
    |   ^(chareType IDENT (^('extends' type))? (^('implements' type+))? classScopeDeclaration*)
        -> {emitCC()}? classDeclaration_cc(
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitCI()}? charedeclaration_ci(
                chareType={$chareType.st},
                arrayDim={null},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        -> {emitH()}? classDeclaration_h(
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        ->
    |   ^('chare_array' ARRAY_DIMENSION IDENT (^('extends' type))? (^('implements' type+))? classScopeDeclaration*)
        -> {emitCI()}? charedeclaration_ci(
                chareType={"array"},
                arrayDim={$ARRAY_DIMENSION.text.toUpperCase()},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds})
        ->
    ;

chareType
@init {
$st = %{$start.getText()};
}
    :   'chare'
    |   'group'
    |   'nodegroup'
    ;

enumConstant
    :   ^(IDENT arguments?)
        -> template(t={$text}) "/*enumConstant-not implemented*/ <t>"
    ;

classScopeDeclaration
@init { boolean entry = false; }
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
        { 
            if ($m.st != null) {
                // determine whether this is an entry method
                entry = listContainsToken($m.start.getChildren(), ENTRY);
            }
        }
        -> {emitCC()}? funcMethodDecl_cc(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        -> {emitH()}? funcMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        -> {(emitCI() && entry)}? funcMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                block={$b.st})
        ->
    |   ^(VOID_METHOD_DECL m=modifierList? g=genericTypeParameterList? IDENT 
            f=formalParameterList b=block?)
        { 
            // determine whether this is an entry method
            if ($m.st != null) {
                entry = listContainsToken($m.start.getChildren(), ENTRY);
            }
        }
        -> {emitCC()}? voidMethodDecl_cc(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitCI() && entry}? voidMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitH()}? voidMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        ->
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
        -> {emitCC() || emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
        -> {emitCC() || emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList b=block)
        { 
            // determine whether this is an entry method
            if ($m.st != null) {
                entry = listContainsToken($m.start.getChildren(), ENTRY);
            }
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
    |   ^(VOID_METHOD_DECL modifierList? genericTypeParameterList? IDENT formalParameterList)
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
    :   ^(MODIFIER_LIST (m+=modifier)+)
        -> mod_list(mods={$m})
    ;

modifier
@init {
$st = %{$start.getText()};
}
    :   'public'
    |   'protected'
    |   'private'
    |   'entry'
    |   'abstract'
    |   'native'
    |   localModifier
        -> {$localModifier.st}
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST (m+=localModifier)+)
        -> local_mod_list(mods={$m})
    ;

localModifier
@init {
$st = %{$start.getText()};
}
    :   'final'
    |   'static'
    |   'volatile'
    ;

    
type
    :   simpleType
        -> {$simpleType.st}
    |   objectType 
        -> {$objectType.st}
    ;

simpleType
    :   ^(TYPE primitiveType arrayDeclaratorList?)
        -> type(typeID={$primitiveType.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

objectType
    :   ^(TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT (t+=typeIdent)+) 
        -> template(types={$t}) "<types; separator=\".\">"
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
        -> typeIdent(typeID={$IDENT.text}, generics={$genericTypeArgumentList.st})
    ;

primitiveType
@init {
$st = %{$start.getText()};
}
    :   'boolean'
        -> template() "bool"
    |   'char'
    |   'byte'
        -> template() "char"
    |   'short'
    |   'int'
    |   'long'
    |   'float'
    |   'double'
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
    |   ^('.' qualifiedIdentifier IDENT)
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
    :   block
        -> {$block.st}
    |   ^('assert' cond=expression msg=expression?)
        -> assert(cond={$cond.st}, msg={$msg.st})
    |   ^('if' parenthesizedExpression then=statement else_=statement?)
        -> if(cond={$parenthesizedExpression.st}, then={$then.st}, else_={$else_.st})
    |   ^('for' forInit cond=expression? (update+=expression)* s=statement)
        -> for(initializer={$forInit.st}, cond={$cond.st}, update={$update}, body={$s.st})
    |   ^(FOR_EACH localModifierList? type IDENT expression statement) 
        -> template(t={$text}) "/* foreach not implemented */ <t>"
    |   ^('while' pe=parenthesizedExpression s=statement)
        -> while(cond={$pe.st}, body={$s.st})
    |   ^('do' s=statement pe=parenthesizedExpression)
        -> dowhile(cond={$pe.st}, block={$s.st})
    |   ^('switch' pe=parenthesizedExpression (scls+=switchCaseLabel)*)
        -> switch(expr={$pe.st}, labels={$scls})
    |   ^('return' e=expression?)
        -> return(val={$e.st})
    |   ^('throw' expression)
        -> template(t={$text}) "/* throw not implemented */ <t>"
    |   ^('break' IDENT?)
        -> template() "break;" // TODO: support labeling
    |   ^('continue' IDENT?)
        -> template() "continue;" // TODO: support labeling
    |   ^(LABELED_STATEMENT i=IDENT s=statement)
        -> label(text={$i.text}, stmt={$s.st})
    |   expression
        -> template(expr={$expression.st}) "<expr>;"
    |   ^('embed' STRING_LITERAL EMBED_BLOCK)
        ->  embed_cc(str={$STRING_LITERAL.text}, blk={$EMBED_BLOCK.text})
    |   ';' // Empty statement.
        -> {%{$start.getText()}}
    ;
        
switchCaseLabel
    :   ^('case' expression (b+=blockStatement)*)
        -> case(expr={$expression.st}, block={$b})
    |   ^('default' (b+=blockStatement)*)
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
    :   ^('=' e1=expr e2=expr)
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
    |   ^('|' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> | <e2>"
    |   ^('^' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> ^ <e2>"
    |   ^('&' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> & <e2>"
    |   ^('==' e1=expr e2=expr)
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
    |   ^('~' e1=expr)
        -> template(e1={$e1.st}) "~<e1>"
    |   ^('!' e1=expr)
        -> template(e1={$e1.st}) "!<e1>"
    |   ^(CAST_EXPR ty=type e1=expr)
        -> template(ty={$ty.st}, e1={$e1.st}) "(<ty>)<e1>"
    |   primaryExpression
        -> {$primaryExpression.st}
    ;
    
primaryExpression
    :   ^('.' prim=primaryExpression IDENT)
        -> template(id={$IDENT}, prim={$prim.st}) "<prim>.<id>"
    |   ^('.' prim=primaryExpression 'this')
        -> template(prim={$prim.st}) "<prim>.this"
    |   ^('.' prim=primaryExpression 'super')
        -> template(prim={$prim.st}) "<prim>.super"
    |   parenthesizedExpression
        -> {$parenthesizedExpression.st}
    |   IDENT
        -> {%{$start.getText()}}
    |   ^(METHOD_CALL pe=primaryExpression gtal=genericTypeArgumentList? args=arguments)
        -> method_call(primary={$pe.st}, generic_types={$gtal.st}, args={$args.st})
    |   explicitConstructorCall
        -> {$explicitConstructorCall.st}
    |   ^(ARRAY_ELEMENT_ACCESS pe=primaryExpression ex=expression)
        -> template(pe={$pe.st}, ex={$ex.st}) "<pe>[<ex>]"
    |   literal
        -> {$literal.st}
    |   newExpression
        -> {$newExpression.st}
    |   'this'
        -> {%{$start.getText()}}
    |   arrayTypeDeclarator
        -> {$arrayTypeDeclarator.st}
    |   'super'
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
    |   'true'
    |   'false'
    |   'null'
    ;

