#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IMaxRedn
#define GENERIC_REDOP_NAME Rmax 
#define GENERIC_OPERATOR  if (x[i] > *y) *y = x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 if (x[i] > y[i]) y[i] = x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IMinRedn
#define GENERIC_REDOP_NAME Rmin
#define GENERIC_OPERATOR  if (x[i] < *y) *y = x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 if (x[i] < y[i]) y[i] = x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME ISumRedn
#define GENERIC_REDOP_NAME Rsum
#define GENERIC_OPERATOR  *y += x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 y[i] += x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IProdRedn
#define GENERIC_REDOP_NAME Rprod
#define GENERIC_OPERATOR  *y *= x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 y[i] *= x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME ICountRedn
#define GENERIC_REDOP_NAME Rcount
#define GENERIC_OPERATOR  (*y) += (x[i]==0) ? 0 : 1;
#define GENERIC_OPERATOR1 y[i]  = (x[i]==0) ? 0 : 1;
#define GENERIC_OPERATOR2 y[i] += (x[i]==0) ? 0 : 1;

