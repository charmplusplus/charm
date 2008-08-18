package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

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

    public String toString() {
        String s = super.toString();
        if (symbol != null) {
            s += "(" + symbol + ")" ;
        }
        return s;
    }

}
