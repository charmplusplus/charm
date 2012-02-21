
package charj.translator;

import java.util.*;

public class CFGNode
{
    public List<CharjAST> statements = new ArrayList<CharjAST>();
    public List<CFGNode> successors = new ArrayList<CFGNode>();
    public List<CFGNode> predecessors = new ArrayList<CFGNode>();
}
