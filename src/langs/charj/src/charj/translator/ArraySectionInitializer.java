package charj.translator;

import org.antlr.stringtemplate.*;
import java.util.ArrayList;

public class ArraySectionInitializer
{
	private static int count = 0; // counts how many instances of this class have been created
	private ArrayList<ArrayList<Object>> ranges;
	private String classType;

	public static int getCount()
	{
		return count;
	}

	public ArraySectionInitializer(ArrayList<ArrayList<Object>> ranges, String classType)
	{
		count++;
		this.ranges = ranges;
		this.classType = classType;
	}

	public String emitCI()
	{
		return "";
	}

	public String emitH()
	{
		return "CProxySection_" + classType + " arraySectionInitializer" + count + "(CProxy_" + classType + " proxyObject)";
	}

	public String emitCC()
	{
		ArrayList<String> indicies = new ArrayList<String>();

		StringTemplate st = new StringTemplate("$signature$\n{\n\t$forLoop$\n}\nreturn CProxySection_$type$::ckNew(proxyObject, elems.getVec(), elems.size());");

		st.setAttribute("signature", emitH());
		st.setAttribute("forLoop", emitCC(indicies, 0));
		st.setAttribute("type", classType);

		return st.toString();
	}
	
	public String emitCC(ArrayList<String> indicies, int dim)
	{
		if(dim == ranges.size())
		{
			String ind = "";
			if(indicies.size() >= 1)
			{
				ind += indicies.get(0);
				for(int i=0; i<indicies.size(); i++)
					ind += (", " + indicies.get(1));
			}

			StringTemplate st = new StringTemplate("elems.push_back(CkArrayIndex$DIM$D($indices$));");
			st.setAttribute("DIM", ranges.size());
			st.setAttribute("indices", ind);
			return st.toString();
		}

		StringTemplate st = new StringTemplate("for(int $coord$ = $start$, $coord$ < $end$, $coord$ += $step$)\n\t$body$\n");

		ArrayList<Object> range = ranges.get(dim);

		Object start = range.get(0);
		Object end = range.get(1);
		Object step = range.size() < 3 ? 1 : range.get(2);

		st.setAttribute("start", start);
		st.setAttribute("end", end);
		st.setAttribute("step", step);

		indicies.add("coord" + dim);

		st.setAttribute("body", emitCC(indicies, ++dim));

		return st.toString();
	}
}




