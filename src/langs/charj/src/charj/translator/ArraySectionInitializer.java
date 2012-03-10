package charj.translator;

import org.antlr.stringtemplate.*;
import java.util.ArrayList;

public class ArraySectionInitializer
{
	private static int count = 0; // counts how many instances of this class have been created
	private ArrayList<ArrayList<Object>> ranges;
	private String classType;
	private String methodName;

	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("ranges: " + ranges + "\n");
		sb.append("classType: " + classType + "\n");
		sb.append("methodName: " + methodName);
		return sb.toString();
	}

	public static int getCount()
	{
		return count;
	}

	public ArraySectionInitializer(ArrayList<ArrayList<Object>> ranges, String classType)
	{
		methodName = "arraySectionInitializer" + (count++);
		this.ranges = ranges;
		this.classType = classType;
	}

	public int getDimensions()
	{
		return ranges.size();
	}

	public String getClassType()
	{
		return classType;
	}

	public String getMethodName()
	{
		return methodName;
	}

	public String emitCI()
	{
		return "";
	}

	public String emitH()
	{
		return "CProxySection_" + classType + " " + methodName + "(CProxy_" + classType + " proxyObject)";
	}

	public String getForLoop()
	{
		ArrayList<String> indicies = new ArrayList<String>();
		/*
		StringTemplate st = new StringTemplate("$signature$\n{\n\t$forLoop$\n}\nreturn CProxySection_$type$::ckNew(proxyObject, elems.getVec(), elems.size());");

		st.setAttribute("signature", emitH());
		st.setAttribute("forLoop", emitCC(indicies, 0));
		st.setAttribute("type", classType);

		return st.toString();
		*/
		return emitCC(indicies, 0);
	}
	
	private String emitCC(ArrayList<String> indicies, int dim)
	{
		if (dim >= ranges.size()) {
			String ind = "";
			if (indicies.size() >= 1) {
				ind += indicies.get(0);
				for(int i = 1; i < indicies.size(); i++)
					ind += (", " + indicies.get(1));
			}

			StringTemplate st = new StringTemplate("elems.push_back(CkArrayIndex$DIM$D($indices$));");
			st.setAttribute("DIM", ranges.size());
			st.setAttribute("indices", ind);
			return st.toString();
		} else {
			StringTemplate st = new StringTemplate("for(int $coord$ = $start$; $coord$ < $end$; $coord$ += $step$)\n\t$body$\n");
			ArrayList<Object> range = ranges.get(dim);

			Object start = range.get(0);
			Object end = range.get(1);
			Object step = range.size() < 3 ? 1 : range.get(2);
			String coord = "coord" + dim;

			st.setAttribute("start", start);
			st.setAttribute("end", end);
			st.setAttribute("step", step);
			st.setAttribute("coord", coord);
			indicies.add(coord);
			st.setAttribute("body", emitCC(indicies, dim + 1));
			return st.toString();
		}
	}
}




