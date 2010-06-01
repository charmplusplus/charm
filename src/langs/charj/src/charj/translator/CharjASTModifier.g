/**
 *  fill a description in
 */

tree grammar CharjASTModifier;

options {
    backtrack = true; 
    memoize = true;
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    output = AST;
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
    Translator translator;

    AstModifier astmod = new AstModifier();
}

// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}


// Starting point for parsing a Charj file.
charjSource[SymbolTable _symtab] returns [ClassSymbol cs]
@init {
    symtab = _symtab;
}
    // TODO: go back to allowing multiple type definitions per file, check that
    // there is exactly one public type and return that one.
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclarations) 
        (typeDeclaration[$importDeclarations.packageNames])*)
        { $cs = $typeDeclaration.sym; }
    ;

packageDeclaration
    :   ^(PACKAGE (ids+=IDENT)+)  
    ;
    
importDeclarations returns [List<CharjAST> packageNames]
    :   (^(IMPORT qualifiedIdentifier '.*'?))*
    ;

typeDeclaration[List<CharjAST> imports] returns [ClassSymbol sym]
    :   ^(TYPE (CLASS | chareType) IDENT (^('extends' parent=type))? (^('implements' type+))? classScopeDeclaration*)
        {
            $TYPE.tree.addChild(astmod.getPupRoutineNode());
            $TYPE.tree.addChild(astmod.getInitRoutineNode());
            
            astmod.ensureDefaultCtor($TYPE.tree);
        }
    |   ^(INTERFACE IDENT (^('extends' type+))?  interfaceScopeDeclaration*)
    |   ^(ENUM IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    ;

chareType
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

enumConstant
    :   ^(IDENT arguments?)
    ;
    
classScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
        {
            if($m.tree == null)
                astmod.fillPrivateModifier($FUNCTION_METHOD_DECL.tree);
        }
    |   ^(PRIMITIVE_VAR_DECLARATION m = modifierList? simpleType variableDeclaratorList)
        {
            if($m.tree == null)
                astmod.fillPrivateModifier($PRIMITIVE_VAR_DECLARATION.tree);
        }
    |   ^(OBJECT_VAR_DECLARATION m = modifierList? objectType variableDeclaratorList)
        {
            if($m.tree == null)
                astmod.fillPrivateModifier($OBJECT_VAR_DECLARATION.tree);
        }
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
        {
            astmod.checkForDefaultCtor($CONSTRUCTOR_DECL);
            if($m.tree == null)
                astmod.fillPrivateModifier($CONSTRUCTOR_DECL.tree);
        }
    ;
    
interfaceScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? 
            type IDENT formalParameterList arrayDeclaratorList?)
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST variableDeclarator+)
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
    ;
    
variableDeclaratorId
    :   ^(IDENT arrayDeclaratorList?)
        {
            astmod.varPup($IDENT);
        }
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
    ;

genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
    ;

genericTypeParameter
    :   ^(IDENT bound?)
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
    ;

modifierList
    :   ^(MODIFIER_LIST modifier+)
        {
            astmod.arrangeModifiers($MODIFIER_LIST.tree);
        }
    ;

modifier
    :   PUBLIC
    |   PRIVATE
    |   PROTECTED
    |   ENTRY
    |   ABSTRACT
    |   NATIVE
    |   localModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

localModifier
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

type
    :   simpleType
    |   objectType 
    |   VOID
    ;

simpleType
    :   ^(SIMPLE_TYPE primitiveType arrayDeclaratorList?)
    ;
    
objectType
    :   ^(OBJECT_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(PROXY_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(POINTER_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
    ;

primitiveType
    :   BOOLEAN     { $start.symbol = new Symbol(symtab, "bool_primitive", symtab.resolveBuiltinType("bool")); }
    |   CHAR        { $start.symbol = new Symbol(symtab, "char_primitive", symtab.resolveBuiltinType("char")); }
    |   BYTE        { $start.symbol = new Symbol(symtab, "byte_primitive", symtab.resolveBuiltinType("char")); }
    |   SHORT       { $start.symbol = new Symbol(symtab, "short_primitive", symtab.resolveBuiltinType("short")); }
    |   INT         { $start.symbol = new Symbol(symtab, "int_primitive", symtab.resolveBuiltinType("int")); }
    |   LONG        { $start.symbol = new Symbol(symtab, "long_primitive", symtab.resolveBuiltinType("long")); }
    |   FLOAT       { $start.symbol = new Symbol(symtab, "float_primitive", symtab.resolveBuiltinType("float")); }
    |   DOUBLE      { $start.symbol = new Symbol(symtab, "double_primitive", symtab.resolveBuiltinType("double")); }
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
    ;
    
genericTypeArgument
    :   type
    |   '?'
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl* formalParameterVarargDecl?) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId)
    ;
    
// FIXME: is this rule right? Verify that this is ok, I expected something like:
// IDENT (^(DOT qualifiedIdentifier IDENT))*
qualifiedIdentifier
    :   IDENT
    |   ^(DOT qualifiedIdentifier IDENT)
    ;
    
block
    :   ^(BLOCK (blockStatement)*)
    ;
    
blockStatement
    :   localVariableDeclaration
    |   statement
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList)
    ;

statement
    :   nonBlockStatement
    |   block
    ;

nonBlockStatement
    :   ^(ASSERT expression expression?)
    |   ^(IF parenthesizedExpression block block?)
    |   ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
    |   ^(FOR_EACH localModifierList? type IDENT expression block) 
    |   ^(WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled break not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(CONTINUE IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled continue not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^('delete' qualifiedIdentifier)
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
    |   ';' // Empty statement.
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
    |   ^(DEFAULT blockStatement*)
    ;
    
forInit
    :   localVariableDeclaration 
    |   expression+
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   ^(PAREN_EXPR expression)
    ;
    
expression
    :   ^(EXPR expr)
    ;

expr
    :   ^(ASSIGNMENT expr expr)
    |   ^(PLUS_EQUALS expr expr)
    |   ^(MINUS_EQUALS expr expr)
    |   ^(TIMES_EQUALS expr expr)
    |   ^(DIVIDE_EQUALS expr expr)
    |   ^(AND_EQUALS expr expr)
    |   ^(OR_EQUALS expr expr)
    |   ^(POWER_EQUALS expr expr)
    |   ^(MOD_EQUALS expr expr)
    |   ^('>>>=' expr expr)
    |   ^('>>=' expr expr)
    |   ^('<<=' expr expr)
    |   ^('?' expr expr expr)
    |   ^(OR expr expr)
    |   ^(AND expr expr)
    |   ^(BITWISE_OR expr expr)
    |   ^(POWER expr expr)
    |   ^(BITWISE_AND expr expr)
    |   ^(EQUALS expr expr)
    |   ^(NOT_EQUALS expr expr)
    |   ^(INSTANCEOF expr type)
    |   ^(LTE expr expr)
    |   ^(GTE expr expr)
    |   ^('>>>' expr expr)
    |   ^('>>' expr expr)
    |   ^(GT expr expr)
    |   ^('<<' expr expr)
    |   ^(LT expr expr)
    |   ^(PLUS expr expr)
    |   ^(MINUS expr expr)
    |   ^(TIMES expr expr)
    |   ^(DIVIDE expr expr)
    |   ^(MOD expr expr)
    |   ^(UNARY_PLUS expr)
    |   ^(UNARY_MINUS expr)
    |   ^(PRE_INC expr)
    |   ^(PRE_DEC expr)
    |   ^(POST_INC expr)
    |   ^(POST_DEC expr)
    |   ^(TILDE expr)
    |   ^(NOT expr)
    |   ^(CAST_EXPR type expr)
    |   primaryExpression
    ;
    
primaryExpression
    :   ^(DOT primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
        ->   ^(ARROW primaryExpression
                   IDENT?
                   THIS?
                   SUPER?
             )
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   ^(ENTRY_METHOD_CALL ^(AT primaryExpression IDENT) genericTypeArgumentList? arguments)
        ->  ^(ENTRY_METHOD_CALL ^(DOT primaryExpression IDENT) genericTypeArgumentList? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
    |   literal
    |   newExpression
    |   THIS
    |   arrayTypeDeclarator
    |   SUPER
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression
    :   ^(  STATIC_ARRAY_CREATOR
            (   primitiveType newArrayConstruction
            |   genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction
            )
        )
    |   ^(NEW qualifiedTypeIdent arguments)
    ;

newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
    |   expression+ arrayDeclaratorList?
    ;

arguments
    :   ^(ARGUMENT_LIST expression*)
    ;

literal 
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

