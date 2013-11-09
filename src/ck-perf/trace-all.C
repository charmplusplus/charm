extern void _createTraceprojections(char **argv);
extern void _createTracesummary(char **argv);
extern void _createTraceprojector(char **argv);
extern void _createTraceautoPerf(char **argv);

void _createTraceall(char **argv)
{
  _createTraceprojections(argv);
  _createTracesummary(argv);
  _createTraceautoPerf(argv);
  _createTraceprojector(argv);
}

