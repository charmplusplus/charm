header {
//header
//Pass2:
// strip-mine MSA for loops
package jade;
}

{
//class preamble
import jade.JJ.J;
import jade.JJ.ASTJ;
import antlr.collections.ASTEnumeration;
}

/** Java 1.3 AST Recognizer Grammar
 *
 * Author: (see java.g preamble)
 * Author: J. DeSouza
 *
 */
class JavaTreeParser2 extends TreeParser;

options {
	importVocab = JavaTreeParser1;
    buildAST = true;
    ASTLabelType = "ASTJ";
}

compilationUnit
	:	(p:packageDefinition)?
		(importDefinition)*
		(typeDefinition[p_AST])*
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
                if ( ((ASTJ)o).hasMain() ) {
                    ((ASTJ)c_AST).status = true; // is a mainchare
                    ((ASTJ)parent).status = true; // is a mainmodule
                }
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
			|	variableDef[true]
			)*
		)
	;

objBlock
	:	#(	OBJBLOCK
			(	ctorDef
			|	methodDef
			|	variableDef[true]
			|	typeDefinition[null]
			|	#(STATIC_INIT slist)
			|	#(INSTANCE_INIT slist)
			)*
		)
	;

ctorDef
	:	#(CTOR_DEF modifiers
            { J.startBlock(); }
            methodHead
            {J.tmp.push("");}
            ctorSList
            {
                J.endBlock();
                J.tmp.pop();
            })
	;

methodDecl
	:	#(METHOD_DEF modifiers typeSpec { J.startBlock(); } methodHead { J.endBlock(); })
	;

methodDef
	:	#(METHOD_DEF modifiers typeSpec
            { J.startBlock(); }
            mh:methodHead
            {
                J.tmp.push(new String(mh.getText()));
            }
            (slist)?
            {
                J.tmp.pop();
                J.endBlock();
            })
	;

variableDef[boolean classVarq]
	:	#(v:VARIABLE_DEF m:modifiers typeSpec 
            vd:variableDeclarator
            varInitializer
            {
                String varName = J.printVariableDeclarator(vd);
                if (!classVarq){
                    J.localStackPush(varName, v);
                }
            })
	;

parameterDef
	:	#(p:PARAMETER_DEF modifiers typeSpec i:IDENT {
                    J.localStackPush(i.getText(), p);
            })
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
	|	variableDef[false]
	|	expression
	|	#(LABELED_STAT IDENT stat)
	|	#("if" expression stat (stat)? )
	|	!#(	fo:"for"
			#(FOR_INIT (variableDef[false] | elist)?)
			#(FOR_CONDITION (expression)?)
			#(FOR_ITERATOR (elist)?)
			st:stat
            {
                AST msa=null;
                // @@ skip for-analysis until it is ready for checkin
                if ( false && (msa=J.isMSAAccessAnywhere(fo))!=null) {
                    // assume we have "for(i = n0; i <= n1; i++) x += A[i];"
                    // need to know: i, n0, n1, A, typeof A
                    // generate: p, pagesize, a
                    // for (p = n0/pagesize; p <= n1/pagesize; p++) {
                    //   double *a = (double*)GetPageBaseAddress(A, p);
                    //   for (i= ((p==n0/pagesize)?n0%p:0); i <= ((p==n1/pagesize)?n1%pagesize:pagesize) )
                    //     x += a[i];
                    // }
                    System.out.print("Found MSA loop: ");
                    System.out.println(fo.toStringTree(0));

                    J.Env e = new J.Env();
                    e.put("arr", msa.getText());
                    J.analyzeFor(fo, e);
                    if (e.get("i2") != null) { // two-level loop

                        // Set up some string variables to make code gen easier.
                        String i = e.getStr("i1");
                        String i0 = e.getStr("n01");  String iFinal = e.getStr("n11"); String iDelta = "1"; // @@
                        String iOP = e.getStr("OP1");
                        if (!iOP.equalsIgnoreCase("<="))
                            J.fatalError(fo, "strip-mining only supports <= in for-condition outer loop. "+iOP);
                        String j = e.getStr("i2");
                        String j0 = e.getStr("n02");  String jFinal = e.getStr("n12"); String jDelta = "1"; // @@
                        String jOP = e.getStr("OP2");
                        if (!jOP.equalsIgnoreCase("<="))
                            J.fatalError(fo, "strip-mining only supports <= in for-condition inner loop. "+jOP);
                        String arr = e.getStr("arr");
                        String arrType = "double"; // @@ hardcoded

                        // Code generation
                        StringBuffer s = new StringBuffer("");

                        s.append("{\n");
                        s.append("int accessMode=-1; int accessPattern=-1;\n");
                        s.append("int MAJOR_SIZE=("+arr+".getArrayLayout() == MSA_ROW_MAJOR)?"+arr+".getCols():"+arr+".getRows();\n");
                        s.append("int numEntriesPerPage="+arr+".getNumEntriesPerPage();\n");
                        s.append("int i0="+i0+"; int iFinal="+iFinal+"; int iDelta="+iDelta+";\n");
                        s.append("int j0="+j0+"; int jFinal="+jFinal+"; int jDelta="+jDelta+";\n");
                        s.append("\n");
                        s.append("int _i = i0; int _j = j0;\n");
                        s.append("do {\n");
                        s.append("int index = ("+arr+".getArrayLayout() == MSA_ROW_MAJOR)?"+arr+".getIndex(_i,_j):"+arr+".getIndex(_j,_i);\n"); // @@ should be accessPattern
                        s.append("int indexSt = (index/numEntriesPerPage)*numEntriesPerPage;\n");
                        s.append(arrType+" pi = addressof("+arr+".getPageBottom(index,Write_Fault));\n"); // @@ pointer, addressof, accessMode
                        s.append("int indexEnd = indexSt + numEntriesPerPage -1;\n");
                        s.append("int iSt = indexSt/MAJOR_SIZE;\n");
                        s.append("int iN = indexEnd/MAJOR_SIZE;\n");
                        s.append("int jSt= indexSt % MAJOR_SIZE;\n");
                        s.append("int jN= indexEnd % MAJOR_SIZE;\n");
                        s.append("\n");
                        s.append("for (; _i"+iOP+"_JADE_MIN(iFinal, iN); _i+="+iDelta+") {\n");
                        s.append("int jEnd;\n");
                        s.append("if (_i==iN) jEnd = _JADE_MIN(jFinal, jN); else jEnd = jFinal;\n");
                        s.append("if (_j>jFinal) _j = j0;\n");
                        s.append("\n");
                        s.append("for(; _j"+jOP+"jEnd; _j+=jDelta) {\n");
                        s.append(arrType+" newname = addressof(pi[((_i-iSt)*MAJOR_SIZE+_j-jSt)]);\n"); // @@ pointer, addressof
                        s.append(i+"=_i;"+j+"=_j;\n");
                        s.append("{ body |= body; }\n"); // body of user loop
                        s.append("}\n");
                        s.append("if (_i==iN && _j>jN && _j<=jFinal)\n");
                        s.append("break;\n");
                        s.append("}\n");
                        s.append("\n");
                        s.append("if (_j>jFinal) _j = j0;\n");
                        s.append("} while ( (_i<iFinal) || ( (_i==iFinal) && (_j<=jFinal) ));\n");
                        s.append("}\n");

//                         s.append("{ int i1; int n01 = "+e.getStr("n01")+"; int n11 = "+e.getStr("n11")+";\n");
//                         s.append("int startPg1 = n01/pageSize; int endPg1 = n11/pageSize;\n");
//                         s.append("int i2; int n02 = "+e.getStr("n02")+"; int n12 = "+e.getStr("n12")+";\n");
//                         s.append("int startPg2 = n02/pageSize; int endPg2 = n12/pageSize;\n");
//                         s.append("for(p1=startPg1; p1<=endPg1; p1++)\n");
//                         s.append("for(p2=startPg2; p2<=endPg2; p2++) {\n");
//                         s.append("double newname[] = A.get(p1*pageSize, p2*pageSize);\n");
//                         s.append("for (i1=(p1==startPg1?(n01)%pageSize:0); i1" + e.get("OP1")
//                             + "(p1==endPg1?n11%pageSize:pageSize); i1++)\n");
//                         s.append("for (i2=(p2==startPg2?(n02)%pageSize:0); i2" + e.get("OP2")
//                             + "(p2==endPg2?n12%pageSize:pageSize); i2++)\n");
//                         s.append("{}}}\n");

                        System.out.println(s);
                        if (true) {
                        // Prepare the new strip-mined code
                        ASTJ newCode = J.parseString(s.toString());
                        newCode.setVerboseStringConversion(true, getTokenNames());
                        System.out.println(newCode.toStringTree(0));
                        newCode.setVerboseStringConversion(false, null);

                        // Find the body of the original code
                        // For a two-level loop, we skip the inner loop
                        ASTJ origBody = st;
                        AST tmp3 = st.getFirstChild();
                        while(tmp3 != null) {
                            origBody = (ASTJ) tmp3;
                            tmp3 = tmp3.getNextSibling();
                        }

                        // Find the location of the body in the new code
                        ASTJ findBody = J.parseString("{ body |= body; }");
                        ASTEnumeration bb = newCode.findAll(findBody);
                        ASTJ theEmptyBody = (ASTJ)bb.nextNode();

                        // Change the original body to use the new array name
                        msa.setText("newname");
                        // Replace the empty new body with the original one
                        theEmptyBody.setFirstChild(astFactory.dupTree(origBody));

//                         AST lastChild = null;
//                         AST lastPointer = null;
//                         AST tmp3 = ttt.getFirstChild();
//                         while(tmp3!=null) {
//                             lastPointer = lastChild;
//                             lastChild = tmp3;
//                             tmp3 = tmp3.getNextSibling();
//                             if (tmp3 == null)
//                                 tmp3 = lastChild.getFirstChild();
//                         }
//                         System.out.println("last child = " + lastChild.toStringTree());

//                         // @@ We assume here that there is a sibling relationship between lastPointer and lastChild,
//                         // which is true in this case since we generate the code.
//                         lastPointer.setNextSibling(astFactory.dupTree(body));
                        stat_AST = newCode;
                        }
                    } else {
                        System.out.println("reached " + fo.toStringTree());
                        stat_AST = fo;
                    }


//                     String s = "{ int i = 0; for (i=0; i<10; i++) ; }";
//                     AST ttt = J.parseString(s);

//                     AST _e1 = #(#[LITERAL_for,"for"],
//                             #(#[FOR_INIT,"FOR_INIT"], #[EXPR,"EXPR"]),
//                             #[FOR_CONDITION,"FOR_CONDITION"],
//                             #[FOR_ITERATOR,"FOR_ITERATOR"],
//                             #[EXPR,"EXPR"]);
//                     System.out.println(fo.toStringTree());
//                     System.out.println();
//                     System.out.println(_e1.toStringTree());
//                     System.out.println(_e1.toStringList());
//                     System.out.println();
                } else {
                    // duplicate the tree, but not its nextChild
                    stat_AST = (ASTJ) astFactory.dupTree(fo);
                }

            }
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
	|	{ J.startBlock(); } slist { J.endBlock(); } // nested SLIST
	|	EMPTY_STAT
	;

caseGroup
	:	#(CASE_GROUP (#("case" expression) | "default")+ slist)
	;

tryBlock
	:	#( "try" slist (handler)* (#("finally" slist))? )
	;

handler
	:	#( "catch" { J.startBlock(); } parameterDef slist { J.endBlock(); } )
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
