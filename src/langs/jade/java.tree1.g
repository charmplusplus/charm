header {
//header
//Pass1:
// build global syntax table
// mark if class is a mainchare
package jade;
}

{
//class preamble
import jade.JJ.J;
import jade.JJ.ASTJ;
}

/** Java 1.3 AST Recognizer Grammar
 *
 * Author: (see java.g preamble)
 * Author: J. DeSouza
 *
 */
class JavaTreeParser1 extends TreeParser;

options {
	importVocab = Java;
    ASTLabelType = "ASTJ";
}

compilationUnit
	:	(p:packageDefinition)?
		(importDefinition)*
		(typeDefinition[p])*
        { if (p != null)
            J.tmp.pop();
        }
	;

packageDefinition
	:	#( PACKAGE_DEF i:identifier { J.tmp.push(J.pE(i)); })
	;

importDefinition
	:	#( IMPORT identifierStar )
	;

typeDefinition[AST parent]
	:	#(c:CLASS_DEF modifiers IDENT { J.tmp.push(#IDENT.getText()); } extendsClause implementsClause
            o:objBlock {
//                 if ( ((ASTJ)o).hasMain() ) {
//                     ((ASTJ)c).status = true; // is a mainchare
//                     ((ASTJ)parent).status = true; // is a mainmodule
//                 }
            }
        ) { J.tmp.pop(); }
	|	#(INTERFACE_DEF modifiers IDENT extendsClause interfaceBlock )
	;

typeSpec
	:	#(TYPE typeSpecArray)
	;

typeSpecArray
	:	#( ARRAY_DECLARATOR typeSpecArray )
    |   #( TEMPLATE typeSpecArray )
	|	type
	;

type:	identifier
	|	builtInType
	;

builtInType
    :   "void"
    |   "boolean"
    |   "byte"
    |   "char"
    |   "short"
    |   "int"
    |   "float"
    |   "long"
    |   "double"
    ;

modifiers
	:	#( MODIFIERS (modifier)* )
	;

modifier
    :   "private"
    |   "public"
    |   "protected"
    |   "static"
    |   "transient"
    |   "final"
    |   "abstract"
    |   "native"
    |   "threadsafe"
    |   "synchronized"
//     |   "const"
    |   "volatile"
	|	"strictfp"
	|	"threaded"
 	|	"blocking"
	|	"readonly"
    ;

extendsClause
	:	#(EXTENDS_CLAUSE (identifier)* )
	;

implementsClause
	:	#(IMPLEMENTS_CLAUSE (identifier)* )
	;

interfaceBlock
	:	#(	OBJBLOCK
			(	methodDecl
			|	variableDef
			)*
		)
	;

objBlock
	:	#(	OBJBLOCK
			(	ctorDef
			|	methodDef
			|	variableDef
			|	typeDefinition[null]
			|	#(STATIC_INIT slist)
			|	#(INSTANCE_INIT slist)
			)*
		)
	;

ctorDef
	:	#(CTOR_DEF modifiers methodHead {J.tmp.push("");} ctorSList {J.tmp.pop();})
	;

methodDecl
	:	#(METHOD_DEF modifiers typeSpec methodHead)
	;

methodDef
	:	#(METHOD_DEF modifiers typeSpec mh:methodHead
            {
                J.globalStackPush(new String(J.fullName(mh.getText())), #METHOD_DEF);
                J.tmp.push(new String(mh.getText()));
            }
            (slist)? { J.tmp.pop(); })
	;

variableDef
	:	#(VARIABLE_DEF m:modifiers typeSpec 
            v:variableDeclarator {
                //System.out.println("pass1: " + v.getText() + J.tmp + "\n");
                if (J.tmp.size() == 2) {
                    J.globalStackPush(new String(J.fullName(v.getText())), #VARIABLE_DEF);
                } else {
                    // should print this only for public static final
                    if ( ((ASTJ)m).isX("static") )
                      J.warning(v, "static variable " + v.getText() + " not accessible outside class.");
                }
            }
            varInitializer)
	;

parameterDef
	:	#(PARAMETER_DEF modifiers typeSpec IDENT )
	;

objectinitializer
	:	#(INSTANCE_INIT slist)
	;

variableDeclarator
	:	IDENT
	|	LBRACK variableDeclarator
	;

varInitializer
	:	#(ASSIGN initializer)
	|
	;

initializer
	:	expression
	|	arrayInitializer
	;

arrayInitializer
	:	#(ARRAY_INIT (initializer)*)
	;

methodHead
	:	IDENT #( PARAMETERS (parameterDef)* ) (throwsClause)?
	;

throwsClause
	:	#( "throws" (identifier)* )
	;

templater
    :  #( TEMPLATE (identifier|constant)+ )
    ;

identifier
	:	#( IDENT (templater)? )
	|	#( DOT identifier #( IDENT (templater)? ) )
	;

identifierStar
	:	IDENT
	|	#( DOT identifier (STAR|IDENT) )
	;

ctorSList
	:	#( SLIST (ctorCall)? (stat)* )
	;

slist
	:	#( SLIST (stat)* )
	;

stat:	typeDefinition[null]
	|	variableDef
	|	expression
	|	#(LABELED_STAT IDENT stat)
	|	#("if" expression stat (stat)? )
	|	#(	"for"
			#(FOR_INIT (variableDef | elist)?)
			#(FOR_CONDITION (expression)?)
			#(FOR_ITERATOR (elist)?)
			stat
		)
	|	#("while" expression stat)
	|	#("do" stat expression)
	|	#("break" (IDENT)? )
	|	#("continue" (IDENT)? )
	|	#("return" (expression)? )
	|	#("switch" expression (caseGroup)*)
	|	#("throw" expression)
	|	#("synchronized" expression stat)
	|	tryBlock
	|	slist // nested SLIST
	|	EMPTY_STAT
	;

caseGroup
	:	#(CASE_GROUP (#("case" expression) | "default")+ slist)
	;

tryBlock
	:	#( "try" slist (handler)* (#("finally" slist))? )
	;

handler
	:	#( "catch" parameterDef slist )
	;

elist
	:	#( ELIST (expression)* )
	;

colonExpression
    :   #(COLON expression expression (expression)?)
    |   expression
    ;

expression
	:	#(EXPR expr)
	;

expr:	#(QUESTION expr expr expr)	// trinary operator
	|	#(ASSIGN expr expr)			// binary operators...
	|	#(PLUS_ASSIGN expr expr)
	|	#(MINUS_ASSIGN expr expr)
	|	#(STAR_ASSIGN expr expr)
	|	#(DIV_ASSIGN expr expr)
	|	#(MOD_ASSIGN expr expr)
	|	#(SR_ASSIGN expr expr)
	|	#(BSR_ASSIGN expr expr)
	|	#(SL_ASSIGN expr expr)
	|	#(BAND_ASSIGN expr expr)
	|	#(BXOR_ASSIGN expr expr)
	|	#(BOR_ASSIGN expr expr)
	|	#(LOR expr expr)
	|	#(LAND expr expr)
	|	#(BOR expr expr)
	|	#(BXOR expr expr)
	|	#(BAND expr expr)
	|	#(NOT_EQUAL expr expr)
	|	#(EQUAL expr expr)
	|	#(LT expr expr)
	|	#(GT expr expr)
	|	#(LE expr expr)
	|	#(GE expr expr)
	|	#(SL expr expr)
	|	#(SR expr expr)
	|	#(BSR expr expr)
	|	#(PLUS expr expr)
	|	#(MINUS expr expr)
	|	#(DIV expr expr)
	|	#(MOD expr expr)
	|	#(STAR expr expr)
	|	#(INC expr)
	|	#(DEC expr)
	|	#(POST_INC expr)
	|	#(POST_DEC expr)
	|	#(BNOT expr)
	|	#(LNOT expr)
	|	#("instanceof" expr expr)
	|	#(UNARY_MINUS expr)
	|	#(UNARY_PLUS expr)
	|	primaryExpression
	;

primaryExpression
    :   IDENT
    |   #(	DOT
			(	expr
				(	IDENT
				|	arrayIndex
				|	"this"
				|	"class"
				|	#( "new" IDENT elist )
				|   "super"
				)
			|	#(ARRAY_DECLARATOR typeSpecArray)
			|	builtInType ("class")?
			)
		)
	|	arrayIndex
	|	#(METHOD_CALL primaryExpression elist)
	|	#(TYPECAST typeSpec expr)
	|   newExpression
	|   constant
    |   "super"
    |   "true"
    |   "false"
    |   "this"
    |   "null"
	|	typeSpec // type name used with instanceof
	;

ctorCall
	:	#( CTOR_CALL elist )
	|	#( SUPER_CTOR_CALL
			(	elist
			|	primaryExpression elist
			)
		 )
	;

arrayIndex
	:	#(INDEX_OP primaryExpression colonExpression)
	;

constant
    :   NUM_INT
    |   CHAR_LITERAL
    |   STRING_LITERAL
    |   NUM_FLOAT
    |   NUM_DOUBLE
    |   NUM_LONG
    ;

newExpression
	:	#(	"new" type
			(	newArrayDeclarator (arrayInitializer)?
			|	elist (objBlock)?
			)
		)
			
	;

newArrayDeclarator
	:	#( ARRAY_DECLARATOR (newArrayDeclarator)? (expression)? )
	;
