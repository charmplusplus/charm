
extern void _createTraceprojections(char **argv);
extern void _createTracesummary(char **argv);

void _createTraceall(char **argv)
{
  _createTraceprojections(argv);
  _createTracesummary(argv);
}

