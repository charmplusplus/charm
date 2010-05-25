package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

class PupRoutineCreator
{
    private CharjAST pupNode;
    private CharjAST block;

    PupRoutineCreator()
    {
        createPupNode();
        block = pupNode.getChild(4);
    }

    protected CharjAST getPupRoutineNode()
    {
        return pupNode;
    }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
    }    

    private void createPupNode()
    {
        pupNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");

    	pupNode.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        pupNode.addChild(createNode(CharjParser.VOID, "void"));
        pupNode.addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        pupNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

        pupNode.getChild(0).addChild(createNode(CharjParser.PUBLIC, "public"));

        pupNode.getChild(3).addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.TYPE, "TYPE"));
        pupNode.getChild(3).getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        pupNode.getChild(3).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "PUP::er&"));

        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

    protected CharjAST getEnclosingType(CharjAST varDeclNode)
    {
        for(CharjAST p = varDeclNode.getParent(); p != null; p = p.getParent())
            if(p.getType() == CharjParser.TYPE)
                return p;
        return null;
    }         

    protected void varPup(CharjAST idNode)
    {
        boolean primitive = false;

        for(CharjAST p = idNode.getParent(); p != null; p = p.getParent())
            if(p.getType() == CharjParser.PRIMITIVE_VAR_DECLARATION)
                primitive = true;
            else if(p.getType() == CharjParser.TYPE)
            {
                if(primitive)
                    primitiveVarPup(idNode);
                else
                    objectVarPup(idNode);
                return;
            }
    }

    protected void primitiveVarPup(CharjAST idNode)
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(0).addChild(createNode(CharjParser.BITWISE_OR, "|"));
        pupNode.getChild(4).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(0).getChild(0).addChild(idNode.dupNode());
    }
    
    protected void objectVarPup(CharjAST varDeclNode){}
}
