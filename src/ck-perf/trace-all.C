
extern void _createTraceprojections(char **argv);
extern void _createTracesummary(char **argv);
extern void _createTraceprojector(char **argv);

void _createTraceall(char **argv)
{
  _createTraceprojections(argv);
  _createTracesummary(argv);
  _createTraceprojector(argv);
}

