/**
 *  fill a description in
 */

tree grammar CharjPreAnalysis;

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
charjSource
    // TODO: go back to allowing multiple type definitions per file, check that
    // there is exactly one public type and return that one.
    :   ^(CHARJ_SOURCE 
        packageDeclaration?
        importDeclaration*
        (externDeclaration
        |readonlyDeclaration
        |typeDeclaration)*)
    ;

packageDeclaration
    :   ^(PACKAGE (ids+=IDENT)+)
    ;
    
importDeclaration
    :   ^(IMPORT qualifiedIdentifier '.*'?)
    ;

readonlyDeclaration
    :   ^(READONLY localVariableDeclaration)
    ;

externDeclaration
    :   ^(EXTERN qualifiedIdentifier)
    ;

typeOfType returns [boolean array_type]
    : CLASS 
    | chareType 
    | chareArrayType { $array_type = true; }
    ;

typeDeclaration
@init {
    astmod = new AstModifier();
}
    :   ^(TYPE typeOfType IDENT
        (^('extends' parent=type))? (^('implements' type+))? classScopeDeclaration*)
        {
        }
        -> ^(TYPE typeOfType IDENT
            (^('extends' type))? (^('implements' type+))? classScopeDeclaration* 
        )
    |   ^(ENUM IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    |   ^(MESSAGE IDENT messageScopeDeclaration*)
    |   ^(MULTICAST_MESSAGE IDENT messageScopeDeclaration*)
    ;

chareArrayType
    :   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

chareType
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    ;

enumConstant
    :   ^(IDENT arguments?)
    ;

messageScopeDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION m = modifierList? simpleType variableDeclaratorList)
        -> {$modifierList.tree != null}? ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
        -> ^(PRIMITIVE_VAR_DECLARATION 
            ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST 'private') ^(LOCAL_MODIFIER_LIST) 
                ^(CHARJ_MODIFIER_LIST) ^(OTHER_MODIFIER_LIST))
            simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION m = modifierList? objectType variableDeclaratorList)
        -> {$modifierList.tree != null}? ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
        -> ^(OBJECT_VAR_DECLARATION  
            ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST 'private') ^(LOCAL_MODIFIER_LIST) 
                ^(CHARJ_MODIFIER_LIST) ^(OTHER_MODIFIER_LIST))
            objectType variableDeclaratorList)
    ;

    
classScopeDeclaration
    :   ^(d=FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=domainExpression? 
            b=block?)
		-> {$m.tree==null}?	^(FUNCTION_METHOD_DECL ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST PRIVATE["private"]) LOCAL_MODIFIER_LIST CHARJ_MODIFIER_LIST OTHER_MODIFIER_LIST)
								genericTypeParameterList? type IDENT formalParameterList domainExpression? block?)
        -> {$m.isEntry}? ^(ENTRY_FUNCTION_DECL modifierList? 
				genericTypeParameterList? type IDENT formalParameterList domainExpression? block?)
        -> ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? 
				type IDENT formalParameterList domainExpression? block?)
    |   ^(DIVCON_METHOD_DECL modifierList? type IDENT formalParameterList divconBlock)
    |   ^(PRIMITIVE_VAR_DECLARATION m = modifierList? simpleType variableDeclaratorList)
        -> {$modifierList.tree != null}? ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
        -> ^(PRIMITIVE_VAR_DECLARATION 
            ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST 'private') ^(LOCAL_MODIFIER_LIST) 
                ^(CHARJ_MODIFIER_LIST) ^(OTHER_MODIFIER_LIST))
            simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION m = modifierList? objectType variableDeclaratorList)
        -> {$modifierList.tree != null}? ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
        -> ^(OBJECT_VAR_DECLARATION  
            ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST 'private') ^(LOCAL_MODIFIER_LIST) 
                ^(CHARJ_MODIFIER_LIST) ^(OTHER_MODIFIER_LIST))
            objectType variableDeclaratorList)
    |   ^(cd=CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            ^(BLOCK (blockStatement*)))

		-> { $m.tree == null }?	^(CONSTRUCTOR_DECL ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST PRIVATE["private"])
																LOCAL_MODIFIER_LIST CHARJ_MODIFIER_LIST OTHER_MODIFIER_LIST)
								genericTypeParameterList? IDENT formalParameterList 
			   		         ^(BLOCK ^(EXPR ^(METHOD_CALL CHELPER ARGUMENT_LIST)) blockStatement*))

        -> { $m.isEntry }? ^(ENTRY_CONSTRUCTOR_DECL modifierList? 
					            genericTypeParameterList? IDENT formalParameterList 
				   		         ^(BLOCK ^(EXPR ^(METHOD_CALL CHELPER ARGUMENT_LIST)) blockStatement*))

        -> ^(CONSTRUCTOR_DECL modifierList? genericTypeParameterList? IDENT formalParameterList 
            ^(BLOCK ^(EXPR ^(METHOD_CALL CHELPER ARGUMENT_LIST)) blockStatement*))
    ;


variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST variableDeclarator+)
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
    ;

variableDeclaratorId
    :   ^(IDENT domainExpression?)
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
    ;

templateArg
    : genericTypeArgument
    | literal
    ;

templateArgList
    :   templateArg templateArg*
    ;

templateInstantiation
    :    ^(TEMPLATE_INST templateArgList)
    |    ^(TEMPLATE_INST templateInstantiation)
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

modifierList returns [boolean isEntry]
    :   ^(MODIFIER_LIST (localModifier | (am+=accessModifier) | charjModifier {if ($charjModifier.isEntry) {$isEntry = true;}} | otherModifier)*)
        -> {$am == null && $isEntry}? ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST PUBLIC["public"]) ^(LOCAL_MODIFIER_LIST localModifier*) ^(CHARJ_MODIFIER_LIST charjModifier*) ^(OTHER_MODIFIER_LIST otherModifier*))
        -> {$am == null}? ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST PRIVATE["private"]) ^(LOCAL_MODIFIER_LIST localModifier*) ^(CHARJ_MODIFIER_LIST charjModifier*) ^(OTHER_MODIFIER_LIST otherModifier*))
        -> ^(MODIFIER_LIST ^(ACCESS_MODIFIER_LIST accessModifier*) ^(LOCAL_MODIFIER_LIST localModifier*) ^(CHARJ_MODIFIER_LIST charjModifier*) ^(OTHER_MODIFIER_LIST otherModifier*)) 
    ;

localModifier
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

accessModifier
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
    ;

charjModifier returns [boolean isEntry] 
    :   ENTRY { $isEntry = true; }
    |   SDAGENTRY { $isEntry = true; }
    |   TRACED
    |   ACCELERATED
    |   THREADED
    ;

otherModifier
    :   ABSTRACT
    |   NATIVE
    ;

modifier
    :   PUBLIC
    |   PRIVATE
    |   PROTECTED
    |   ENTRY
    |   TRACED
    |   ABSTRACT
    |   NATIVE
    |   localModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

type
    :   simpleType
    |   objectType 
    |   VOID
    ;

simpleType
    :   ^(SIMPLE_TYPE primitiveType domainExpression?)
    ;
    
objectType
    :   ^(OBJECT_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(POINTER_TYPE qualifiedTypeIdent domainExpression?)
	|	^(MESSAGE_TYPE qualifiedTypeIdent)
	|	^(ARRAY_SECTION_TYPE qualifiedTypeIdent domainExpression?)
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent
    :   ^(IDENT templateInstantiation?)
    ;

primitiveType
    :   BOOLEAN
    |   CHAR
    |   BYTE
    |   SHORT
    |   INT
    |   LONG
    |   FLOAT
    |   DOUBLE
    ;

genericTypeArgument
    :   type
    |   '?'
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl*) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
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
    |   sdagStatement
    |   block
    ;

divconBlock
    :   ^(DIVCON_BLOCK divconExpr)
    ;

divconAssignment
    :   ^(LET_ASSIGNMENT IDENT expression)
    ;

divconAssignmentList
    :   divconAssignment+
    ;

divconExpr
    :   ^(IF parenthesizedExpression divconExpr divconExpr?)
    |   ^(LET divconAssignmentList IN divconExpr)
    |   expression
    ;

sdagStatement
    :   ^(OVERLAP block)
    |   ^(WHEN (IDENT expression? formalParameterList)* block)
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
    |   ^('delete' expression)
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
	|	^(CONTRIBUTE_1 expression)
	|	^(CONTRIBUTE_2 expression expression qualifiedIdentifier expression)
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
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression templateInstantiation? arguments)
    |   ^(ENTRY_METHOD_CALL ^(AT primaryExpression IDENT) templateInstantiation? arguments)
        ->  ^(ENTRY_METHOD_CALL ^(DOT primaryExpression IDENT) templateInstantiation? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression domainExpression)
    |   literal
    |   newExpression
    |   THIS
    |   arrayTypeDeclarator
    |   SUPER
	|	THISINDEX
	|	THISPROXY
    |   domainExpression
    |   ^(SIZEOF (expression | type))
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL templateInstantiation? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? templateInstantiation? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression
    :   ^(NEW_EXPRESSION arguments? domainExpression)
    |   ^(NEW type arguments)
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

rangeExpression
    :   ^(RANGE_EXPRESSION expression expression? expression?)
    ;

rangeList
    :   rangeExpression+
    ;

domainExpression
    :   ^(DOMAIN_EXPRESSION rangeList)
    ;
