# Tarun Prabhu
# 8 Nov 2010

# Checks if the solutions to a k-SAT problem are correct
# The input file should be in DIMACS format
# The output file should be in the format
#   [One solution:
#    (literal_number=[true|false|1|-1|0])+
#   ]+

# USAGE: soln-checker.py [problemfile] [solutionfile]

import math
import re
import sys

numberRegex = re.compile("[-]?\d+")

# list<int> parseClause (string)
def parseClause (clauseString):
    lits = clauseString.split()
    clause = []
    for lit in lits:
        if (numberRegex.match(lit) and int(lit) != 0):
            clause.append(int(lit))
    return clause

# list<list<int>> parseClauses (list<string>)
def parseClauses (clauseStrings):
    clauses = []
    for clauseString in clauseStrings:
        clause = parseClause(clauseString)
        if (clause != []):
            clauses.append(clause)
    return clauses

# list<string> findClauses (list<string>)
def findClauses (lines):
    clauseLines = []
    for line in lines:
        if (not (line.startswith("c") or line.startswith("p"))):
            clauseLines.append(line)
    return clauseLines

# list<list<int>> readProblemFile (string)
def readProblemFile (problemFileName):
    file = open(problemFileName, "r")
    lines = file.readlines()
    file.close()

    clauses = parseClauses(findClauses(lines))
    return clauses

# list<string> findSolutions (list<string> linesInSolutionFile)
def findSolutions(lines):
    solutions = []
    foundSolutionTag = False
    for line in lines:
        # If the line begins with "One solution", it must be a title line
        # The line immediately after the "One Solution" line is the solution
        # This still assumes that the entire solution is printed
        # on the same line. It's better than before but I'd like it to be a 
        # little more robust if possible.
        if (line.startswith("One solution:")):
            foundSolutionTag = True
        elif (foundSolutionTag):
            solutions.append(line)
            foundSolutionTag = False
    return solutions

# list<list<int>> findMatchingParens(string, int)
# Returns a list of pairs of indices for each "(",")" pair
def findMatchingParens (string):
    matchingParens = []
    startIdx = -1
    endIdx = -1
    for i in range(0, len(string), 1):
        if (string[i] == "("):
            startIdx = i
        elif(string[i] == ")"):
            endIdx = i
            matchingParens.append([startIdx, endIdx])
    return matchingParens


# #dict<int, bool> parseSolution (string solution)
# Returns a map from literal number to its value
def parseSolution (solution):
    map = {}
    matchingParens = findMatchingParens(solution)
    for match in matchingParens:
        start = match[0]
        end = match[1]
        string = solution[(start+1):end]
        (strLit, strVal) = string.split("=")
        if (strVal == "true" or strVal == "1"):
            val = True
        else: #(strVal == "false" or strVal == "0" or strVal == "-1")
            val = False
        map[int(strLit)] = val
    return map

# list<dict<int,bool>> parseSolutions(list<string> solutions)
def parseSolutions(solutions):
    solutionMaps = []
    for solution in solutions:
        solutionMaps.append(parseSolution(solution))
    return solutionMaps

# list<dict<int,bool>> readSolutionFile (string solutionFileName)
# Returns a list of dictionaries where each dictionary represents a solution
def readSolutionFile (solutionFileName):
    file = open(solutionFileName)
    lines = file.readlines()
    file.close()

    solutions = parseSolutions(findSolutions(lines))
    return solutions

# void printSolution (dict<int,bool>)
def printSolution(solution):
    for lit, val in solution.iteritems():
        print "(" + str(lit) + "=" + str(val) + ")",

# void printClauses (list<list<int>>)
def printClauses(clauses):
    for clause in clauses:
        for lit in clause:
            print lit, 
        print ""

# bool checkSolution (list<list<int>>, list<map<int,bool>>)
def checkSolution(clauses, solution):
    solutionResult = True
    for clause in clauses:
        clauseResult = False
        for lit in clause:
            absLit = math.fabs(lit)
            if (lit < 0):
                clauseResult = clauseResult or (not solution[absLit])
            else:
                clauseResult = clauseResult or solution[absLit]
        solutionResult = solutionResult and clauseResult
    return solutionResult
    
# bool checkSolutions (list<list<int>>, list<map<int,bool>>)
def checkSolutions (clauses, solutions):
    allOk = True
    i = 1
    for solution in solutions:
        result = checkSolution(clauses, solution)
#         print "Solution #" + str(i) + ": " + str(result)
        allOk = allOk and result
        i = i + 1
    return allOk

def main():
    if (len(sys.argv) != 3):
        print "USAGE: soln-checker.py [problemfile] [solutionfile]"
    else:
        inputFileName = sys.argv[1]
        outputFileName = sys.argv[2]

        clauses = readProblemFile(inputFileName)
#        printClauses(clauses)
        solutions = readSolutionFile(outputFileName)
#         for solution in solutions:
#             print "One solution:"
#             printSolution(solution)
#             print "\n"
        print "All solutions passed: "+ str(checkSolutions(clauses, solutions))

if __name__ == "__main__":
    main()
