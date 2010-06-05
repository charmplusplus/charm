package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.tree.Tree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;
import java.util.*;

/**
 * Custom subclass of Antlr's tree node. Doesn't do anything special yet,
 * it's just here to make it easier to add features later.
 */
public class CharjAST extends CommonTree
{
    /** Semantic information about this node. Includes type, scope, location
     * of definition, etc. */
    public Symbol symbol;

    public CharjAST(Token t) {
        super(t);
    }

    public CharjAST(int type, String text) {
        super(new CommonToken(type, text));
    }

    public CharjAST getParent()
    {
        return (CharjAST) super.getParent();
    }

    public List<CharjAST> getChildren()
    {
        return (List<CharjAST>) super.getChildren();
    }

    public CharjAST getChild(int index)
    {
        try
        {
            return (CharjAST) super.getChild(index);
        }
        catch(ClassCastException e)
        {
            Tree child = super.getChild(index);
            System.out.println("possible error node: " + child);
            return new CharjAST(child.getType(), child.getText());
        }
    }

    public String toString() {
        String s = super.toString();
        if (symbol != null) {
            s += "(" + symbol + ")" ;
        }
        return s;
    }

    public CharjAST dupNode()
    {
        return new CharjAST(getType(), getText());
    }

    public CharjAST dupTree()
    {
        CharjAST root = dupNode();

        List<CharjAST> children = getChildren();
        if(children == null) return root;

        for(CharjAST child : getChildren())
            root.addChild(child.dupTree());
        return root;
    }

    public CharjAST getChildOfType(int type)
    {
        try
        {
            for(CharjAST c : getChildren())
                if(c.getType() == type)
                    return c;
        }
        catch(NullPointerException npe)
        {
            npe.printStackTrace();
        }
        return null;
    }

    public CharjAST getChildAfterType(int type)
    {
        for(int i = 0; i < getChildCount(); i++)
            if(getChild(i).getType() == type)
                return getChild(i+1);
        return getChild(0);
    }

    @Override
    public boolean equals(Object o)
    {
        if(o == null)
            return false;
        if(!(o instanceof CharjAST))
            return false;
        CharjAST other = (CharjAST)o;
        return other.getType() == this.getType() && other.getText().equals(this.getText());
    }
}
