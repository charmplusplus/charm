/**
 *  TODO add a description
 */

tree grammar CharjPostAnalysis;

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

// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}

@members {
    SymbolTable symtab = null;
    PackageScope currentPackage = null;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
    LocalScope currentLocalScope = null;
    Translator translator;

    AstModifier astmod = new AstModifier();

    protected boolean containsModifier(CharjAST modlist, int type)
    {
        if(modlist == null)
            return false;
        CharjAST charjModList = modlist.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
        if(charjModList == null)
            return false;
        if(charjModList.getChildOfType(CharjParser.ENTRY) == null)
            return false;
        return true;
    }

	String getQualIdText(CharjAST qid)
	{
		StringBuilder sb = new StringBuilder();

		sb.append(qid.getChild(0).getText());

		for(int i = 1; i < qid.getChildren().size(); i++)
			sb.append("::" + qid.getChild(i).getText());

		return sb.toString();
	}
}
	
// Starting point for parsing a Charj file.
charjSource[SymbolTable _symtab] returns [ClassSymbol cs]
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclaration
        | typeDeclaration { $cs = $typeDeclaration.sym; }
        | externDeclaration
        | readonlyDeclaration)*)
    ;

// note: no new scope here--this replaces the default scope
packageDeclaration
    :   ^(PACKAGE IDENT+)
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

typeDeclaration returns [ClassSymbol sym]
    :   ^(TYPE classType IDENT { currentClass = (ClassSymbol) $IDENT.def.type; }
            (^(EXTENDS parent=type))? (^('implements' type+))?
                classScopeDeclaration*)
    |   ^('enum' IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    |   ^(MESSAGE IDENT messageScopeDeclaration*)
    |   ^(MULTICAST_MESSAGE IDENT messageScopeDeclaration*)
    ;

classType
    :   CLASS
    |   chareType
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

messageScopeDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
    ;

classScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList?
            type IDENT formalParameterList domainExpression? b=block)
    |   ^(ENTRY_FUNCTION_DECL modifierList? genericTypeParameterList?
            type IDENT entryFormalParameterList domainExpression? b=block?)
        -> {$b.sdag}? ^(SDAG_FUNCTION_DECL modifierList? genericTypeParameterList?
                type IDENT entryFormalParameterList domainExpression? block?)
        -> ^(ENTRY_FUNCTION_DECL modifierList? genericTypeParameterList?
            type IDENT entryFormalParameterList domainExpression? block?)
    |   ^(DIVCON_METHOD_DECL modifierList? type IDENT formalParameterList divconBlock)
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
            //^(VAR_DECLARATOR_LIST field[$simpleType.type]+))
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
            //^(VAR_DECLARATOR_LIST field[$objectType.type]+))
    |   ^(CONSTRUCTOR_DECL modifierList? genericTypeParameterList? IDENT formalParameterList 
            block)
    |   ^(ENTRY_CONSTRUCTOR_DECL modifierList? genericTypeParameterList? IDENT entryFormalParameterList 
            block)
    ;

field [ClassSymbol type]
    :   ^(VAR_DECLARATOR ^(IDENT domainExpression?) variableInitializer?)
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

modifierList
    :   ^(MODIFIER_LIST accessModifierList localModifierList charjModifierList otherModifierList)
    ;

modifier
    :   accessModifier
    |   localModifier
    |   charjModifier
    |   otherModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier*)
    ;

accessModifierList
    :   ^(ACCESS_MODIFIER_LIST accessModifier*)
    ;

charjModifierList
@init { boolean ctor = false; }
    :   ^(CHARJ_MODIFIER_LIST charjModifier* {
            // Non contructor entry methods are potential reduction targets
            ctor = $CHARJ_MODIFIER_LIST.hasParentOfType(ENTRY_CONSTRUCTOR_DECL);
        })
        -> {ctor}? ^(CHARJ_MODIFIER_LIST charjModifier* )
        -> ^(CHARJ_MODIFIER_LIST charjModifier* REDUCTIONTARGET)
    ;

otherModifierList
    :   ^(OTHER_MODIFIER_LIST otherModifier*)
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

charjModifier
    :   ENTRY
    |   SDAGENTRY
    |   TRACED
    |   ACCELERATED
    |   THREADED
    ;

otherModifier
    :   ABSTRACT
    |   NATIVE
    ;

entryArgType
    :   simpleType
    |   entryArgObjectType
    |   VOID
    ;

type
    :   simpleType
    |   objectType
    |   VOID
    ;

nonArraySectionObjectType returns [Type type]
@after
{
	$type = $start.symbolType;
}
	:	simpleType
	|   ^(OBJECT_TYPE qualifiedTypeIdent domainExpression?)
	|   ^(REFERENCE_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(POINTER_TYPE qualifiedTypeIdent domainExpression?)
	|	^(MESSAGE_TYPE qualifiedTypeIdent)
	|	VOID
	;	

simpleType returns [ClassSymbol type]
    :   ^(SIMPLE_TYPE primitiveType domainExpression?)
    ;
    
objectType returns [ClassSymbol type]
    :   ^(OBJECT_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(POINTER_TYPE qualifiedTypeIdent domainExpression?)
	|	^(MESSAGE_TYPE qualifiedTypeIdent)
	|	^(ARRAY_SECTION_TYPE qualifiedTypeIdent domainExpression?)
    ;

entryArgObjectType returns [ClassSymbol type]
    :   ^(OBJECT_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(POINTER_TYPE qualifiedTypeIdent domainExpression?)
    |   ^(MESSAGE_TYPE qualifiedTypeIdent)
    ;

qualifiedTypeIdent returns [ClassSymbol type]
    :   ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent returns [String name]
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

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
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

entryFormalParameterList
    :   ^(FORMAL_PARAM_LIST entryFormalParameterStandardDecl*) 
    ;

entryFormalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? entryArgType variableDeclaratorId)
    ;
    
// FIXME: is this rule right? Verify that this is ok, I expected something like:
// IDENT (^(DOT qualifiedIdentifier IDENT))*
qualifiedIdentifier
    :   IDENT
    |   ^(DOT qualifiedIdentifier IDENT)
    ;
    
block returns [boolean sdag]
@init { $sdag = false; }
    :   ^(BLOCK (blockStatement { $sdag |= $blockStatement.sdag; })*)
    ;

blockStatement returns [boolean sdag]
@init { $sdag = false; }
    :   localVariableDeclaration
    |   statement { $sdag = $statement.sdag; }
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList)
    ;

statement returns [boolean sdag]
@init { $sdag = false; }
    : nonBlockStatement { $sdag = $nonBlockStatement.sdag; }
    | sdagStatement { $sdag = true; }
    | block { $sdag = $block.sdag; }
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

nonBlockStatement returns [boolean sdag]
@init { $sdag = false; }
    :   ^(ASSERT expression expression?)
    |   ^(IF parenthesizedExpression (i=block { $sdag |= $i.sdag; }) (e=block { $sdag |= $e.sdag; })?)
        -> {$sdag}? ^(SDAG_IF parenthesizedExpression $i $e?)
        -> ^(IF parenthesizedExpression $i $e?)
    |   ^(FOR forInit? FOR_EXPR (e1=expression)? FOR_UPDATE (e2+=expression)* block {
            $sdag = $block.sdag;
        })
        -> {$sdag}? ^(SDAG_FOR forInit? FOR_EXPR $e1? FOR_UPDATE $e2? block)
        -> ^(FOR forInit? FOR_EXPR $e1? FOR_UPDATE $e2? block)
    |   ^(FOR_EACH localModifierList? type IDENT expression block { $sdag = $block.sdag; })
    |   ^(WHILE parenthesizedExpression block { $sdag = $block.sdag; })
        -> {$sdag}? ^(SDAG_WHILE parenthesizedExpression block)
        -> ^(WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression { $sdag = $block.sdag; })
        -> {$sdag}? ^(SDAG_DO block parenthesizedExpression)
        -> ^(DO block parenthesizedExpression)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?) 
    |   ^(CONTINUE IDENT?) 
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
    
primaryExpression returns [Type type]
    :   ^(DOT pe=primaryExpression
                (   IDENT { $type = $IDENT.symbolType; }
                |   THIS  { $type = $IDENT.symbolType; }
                |   SUPER { $type = $IDENT.symbolType; }
                )
        )
		-> { $pe.type instanceof PointerType }? ^(ARROW primaryExpression IDENT? THIS? SUPER?)
		-> { $pe.type instanceof MessageType }? ^(ARROW primaryExpression IDENT? THIS? SUPER?)
		->										^(DOT primaryExpression IDENT? THIS? SUPER?)
    |   parenthesizedExpression
    |   IDENT
		{
			$type = $IDENT.symbolType;
		}
    |   CHELPER
    |   ^(METHOD_CALL pe=primaryExpression genericTypeArgumentList? arguments) 
		{
			$type = $METHOD_CALL.symbolType;
		}
    |   ^(ENTRY_METHOD_CALL pe=primaryExpression genericTypeArgumentList? entryArguments)
		{
			$type = $ENTRY_METHOD_CALL.symbolType;
		}
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS pe=primaryExpression domainExpression)
		{
			$type = $ARRAY_ELEMENT_ACCESS.symbolType; // TODO this is not correct, as it's always null
			if($pe.type instanceof ProxyType && $domainExpression.ranges.get(0).size() > 1)
			{
				//System.out.println("creating a new ArraySectionInitializer");
				ArraySectionInitializer asi = new ArraySectionInitializer($domainExpression.ranges, ((ProxyType)$pe.type).baseType.getTypeName());
				currentClass.sectionInitializers.add(asi);
				//System.out.println(asi);
			}
		}
			->	{ $pe.type instanceof ProxyType && $domainExpression.ranges.get(0).size() > 1 }? ^(METHOD_CALL IDENT["arraySectionInitializer" + (ArraySectionInitializer.getCount() - 1)] ^(ARGUMENT_LIST ^(EXPR primaryExpression)))
			-> 																			 		 ^(ARRAY_ELEMENT_ACCESS primaryExpression domainExpression)
    |   literal
    |   newExpression
		{
			$type = $newExpression.type;
		}
	|	THIS
		{
			$type = $THIS.symbolType;
		}
    |   arrayTypeDeclarator
    |   SUPER
		{
			$type = $SUPER.symbolType;
		}
	|	THISINDEX
	|	THISPROXY
		{
			$type = $THISPROXY.symbolType;
		}
    |   domainExpression
    |   ^(SIZEOF (expression | type))
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression returns [Type type]
    :   ^(NEW_EXPRESSION arguments? domainExpression)
    |   ^(NEW nonArraySectionObjectType arguments)
		{
			$type = $nonArraySectionObjectType.type;
		}
    ;

arguments returns [Object expr]
    :   ^(ARGUMENT_LIST expression*) 
    ;

entryArguments
    :   ^(ARGUMENT_LIST entryArgExpr*)
    ;

entryArgExpr
    :   ^(EXPR entryExpr)
        -> {$EXPR.symbolType instanceof PointerType}? ^(EXPR ^(POINTER_DEREFERENCE entryExpr))
        -> ^(EXPR entryExpr)
    ;

entryExpr
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
    |   entryPrimaryExpression
    ;
    
entryPrimaryExpression
    :   ^(DOT primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   ^(ARROW primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   parenthesizedExpression
    |   IDENT
        {
            //System.out.println("Derefing ID with type " + $IDENT.symbolType.getClass().getName() +
            //    ":\n" + $IDENT.symbolType + "\nand def info " + $IDENT.def.getClass().getName() + ":\n" + $IDENT.def);
            if ($IDENT.symbolType instanceof ClassSymbol) {
                if (!((ClassSymbol)$IDENT.symbolType).isPrimitive) {
                    astmod.makePointerDereference($IDENT);
                }
            }
        }
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   ^(ENTRY_METHOD_CALL primaryExpression genericTypeArgumentList? entryArguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression domainExpression)
    |   literal
    |   newExpression
        ->  ^(POINTER_DEREFERENCE newExpression)
    |   THIS
        ->  ^(POINTER_DEREFERENCE THIS)
    |   arrayTypeDeclarator
    |   SUPER
    |   GETNUMPES
    |   GETNUMNODES
    |   GETMYPE
    |   GETMYNODE
    |   GETMYRANK
    |   domainExpression
    |   ^(SIZEOF (expression | type))
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

rangeItem returns [Object item]
    :   ^(EXPR expr){ $item = $EXPR; }
    ;

rangeExpression returns [ArrayList<Object> range]
@init
{
	$range = new ArrayList<Object>();
}
    :   ^(RANGE_EXPRESSION i1=rangeItem (i2=rangeItem)? (i3=rangeItem)?) {
            $range.add($i1.item);
            if (i2 != null) $range.add($i2.item);
            if (i3 != null) $range.add($i3.item);
        }
    ;

rangeList returns [ArrayList<ArrayList<Object>> ranges]
@init
{
	$ranges = new ArrayList<ArrayList<Object>>();
}
    :   (rangeExpression { $ranges.add($rangeExpression.range); })+
    ;

domainExpression returns [ArrayList<ArrayList<Object>> ranges]
    :   ^(DOMAIN_EXPRESSION rangeList)	{ $ranges = $rangeList.ranges; }
    ;
