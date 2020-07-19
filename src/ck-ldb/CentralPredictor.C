/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <math.h>
#include <charm++.h>
#include "CentralLB.h"

void CentralLB::staticPredictorOn(void* data, void *model)
{
  CentralLB *me = (CentralLB*)(data);
  me->predictorOn((LBPredictorFunction*)model);
}

void CentralLB::staticPredictorOnWin(void* data, void *model, int wind)
{
  CentralLB *me = (CentralLB*)(data);
  me->predictorOn((LBPredictorFunction*)model, wind);
}

void CentralLB::staticPredictorOff(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->predictorOff();
}

void CentralLB::staticChangePredictor(void* data, void *model)
{
  CentralLB *me = (CentralLB*)(data);
  me->changePredictor((LBPredictorFunction*)model);
}

#define MAX_CHISQ_ITER 10000
#define SMALL_NUMBER   0.00001    // to avoid singular matrix in gaussj

void gaussj(double **a, double *b, int n) {
#if CMK_LBDB_ON
  int i,j,k;
  int irow, icol;
  double big, dum, pivinv;
  // arrays for bookkeeping on the pivoting
  int *indxc, *indxr, *ipiv;

  indxc = new int[n];
  indxr = new int[n];
  ipiv = new int[n];
  for (j=0; j<n; ++j) ipiv[j]=0;
  // main loop over the columns to be reduced
  for (i=0; i<n; ++i) {
    big = 0;
    // outer loop of the search for a pivot element
    for (j=0; j<n; ++j)
      if (ipiv[j] != 1)
	for (k=0; k<n; ++k) {
	  if (ipiv[k] == 0 && fabs(a[j][k]) >= big) {
	    big = fabs(a[j][k]);
	    irow=j;
	    icol=k;
	  }
	}
    ++(ipiv[icol]);

    if (irow != icol) {
      for (j=0; j<n; ++j) {dum=a[irow][j]; a[irow][j]=a[icol][j]; a[icol][j]=dum;}
      dum=b[irow]; b[irow]=b[icol]; b[icol]=dum;
    }
    // we are now ready to divide the pivot row by the pivot element, located at irow, icol
    indxr[i]=irow;
    indxc[i]=icol;
    if (a[icol][icol] == 0) {
      a[icol][icol] = SMALL_NUMBER;
      CkPrintf("LB: Singular Matrix\n");
    }
    pivinv = 1.0/a[icol][icol];
    a[icol][icol] = 1;
    for (j=0; j<n; ++j) a[icol][j] *= pivinv;
    b[icol] *= pivinv;
    for (j=0; j<n; ++j)
      if (j != icol) {
	dum = a[j][icol];
	a[j][icol] = 0;
	for (k=0; k<n; ++k) a[j][k] -= a[icol][k]*dum;
	b[j] -= b[icol]*dum;
      }
  }
  // unscramble the matrix
  for (i=n-1; i>=0; --i) {
    if (indxr[i] != indxc[i])
      for (j=0; j<n; ++j) {dum=a[j][indxr[i]]; a[j][indxr[i]]=a[j][indxc[i]]; a[j][indxc[i]]=dum;}
  }
  delete[] indxr;
  delete[] indxc;
  delete[] ipiv;

#endif
}

void Marquardt_coefficients(double *x, double *y, double *param, double **alpha, double *beta, double &chisq, LBPredictorFunction *predict) {
#if CMK_LBDB_ON
  int i,j,k,l,m;
  double ymod, dy;
  double *dyda = new double[predict->num_params];

  for (i=0; i<predict->num_params; ++i) {
    for (j=0; j<=i; ++j) alpha[i][j] = 0;
    beta[i]=0;
  }
  chisq = 0;

  // summation loop over all data
  for (i=0; i<predict->num_params; ++i) {
    predict->function(x[i], param, ymod, dyda);
    dy = y[i] - ymod;
    for (j=0, l=0; l<predict->num_params; ++l) {
      for (k=0, m=0; m<l+1; ++m) {
	alpha[j][k++] += dyda[l]*dyda[m];
      }
      beta[j++] += dy*dyda[l];
    }
    chisq += dy*dy;
  }

  // fill the symmetric side
  for (j=1; j<predict->num_params; ++j) {
    for (k=0; k<j; ++k) alpha[k][j] = alpha[j][k];
  }

  delete[] dyda;
#endif
}

bool Marquardt_solver(CentralLB::FutureModel *mod, int object) {
#if CMK_LBDB_ON
  double chisq, ochisq;
  double lambda = 0.001;
  int i,j;
  int iterations=0;
  bool allow_stop = false;

  double *oneda = new double[mod->predictor->num_params];
  double *atry = new double[mod->predictor->num_params];
  double *beta = new double[mod->predictor->num_params];
  double *da = new double[mod->predictor->num_params];
  double **covar = new double*[mod->predictor->num_params];
  double **alpha = new double*[mod->predictor->num_params];
  double *x = new double[mod->cur_stats-1];
  double *y = new double[mod->cur_stats-1];
  double **temp = new double*[mod->predictor->num_params];

  for (i=0; i<mod->predictor->num_params; ++i) {
    alpha[i] = new double[mod->predictor->num_params];
    covar[i] = new double[mod->predictor->num_params];
    temp[i] = new double[mod->predictor->num_params];
    atry[i] = mod->parameters[object][i];
  }
  for (i=0; i<mod->cur_stats-2; ++i) {
    x[i] = mod->collection[i].objData[object].wallTime;
    y[i] = mod->collection[i+1].objData[object].wallTime;
  }

  Marquardt_coefficients(x,y,mod->parameters[object],alpha,beta,chisq,mod->predictor);
  ochisq = chisq;

  while (chisq > 0.01 || !allow_stop) {
    if (++iterations > MAX_CHISQ_ITER) {
      // something wrong!!!
      return false;
    }
    // alter linearized fitting matrix, by augmenting diagonal elements
    for (i=0; i<mod->predictor->num_params; ++i) {
      for (j=0; j<mod->predictor->num_params; ++j) covar[i][j] = alpha[i][j];
      covar[i][i] = alpha[i][i] * (1 + lambda);
      for (j=0; j<mod->predictor->num_params; ++j) temp[i][j] = covar[i][j];
      oneda[i] = beta[i];
    }

    // matrix solution
    gaussj(temp, oneda, mod->predictor->num_params);
    for (i=0; i<mod->predictor->num_params; ++i) {
      for (j=0; j<mod->predictor->num_params; ++j) covar[i][j] = temp[i][j];
      da[i] = oneda[i];
    }

    // did the trial succeed?
    for (i=0, j=0; j<mod->predictor->num_params; ++j) atry[j] = mod->parameters[object][j] + da[i++];
    Marquardt_coefficients(x,y,atry,covar,da,chisq,mod->predictor);
    if (chisq < ochisq) {  // success, accept the new solution
      lambda *= 0.1;
      allow_stop = true;
      for (i=0; i<mod->predictor->num_params; ++i) {
	for (j=0; j<mod->predictor->num_params; ++j) alpha[i][j] = covar[i][j];
	beta[i] = da[i];
	mod->parameters[object][i] = atry[i];
      }
    } else {  // failure, increase lamda
      lambda *= 10;
      allow_stop = false;
    }
    ochisq = chisq;
  }
  for (i=0; i<mod->predictor->num_params; ++i) {
    delete[] alpha[i];
    delete[] covar[i];
    delete[] temp[i];
  }
  delete[] oneda;
  delete[] atry;
  delete[] beta;
  delete[] da;
  delete[] covar;
  delete[] alpha;
  delete[] x;
  delete[] y;
  delete[] temp;

#endif
  return true;
}

// routine that update LDStats given a predictor model
void CentralLB::FuturePredictor(BaseLB::LDStats* stats) {
#if CMK_LBDB_ON
  bool model_done;
  int i;

  if (predicted_model->cur_stats < _lb_predict_delay) {
    // not yet ready to create the model, just store the relevant statistic
    predicted_model->collection[predicted_model->start_stats].objData.resize(stats->n_objs);
    predicted_model->collection[predicted_model->start_stats].commData.resize(stats->n_comm);
    predicted_model->collection[predicted_model->start_stats].n_objs = stats->n_objs;
    predicted_model->collection[predicted_model->start_stats].n_migrateobjs = stats->n_migrateobjs;
    predicted_model->collection[predicted_model->start_stats].n_comm = stats->n_comm;
    for (i=0; i<stats->n_objs; ++i)
      predicted_model->collection[predicted_model->start_stats].objData[i] = stats->objData[i];
    for (i=0; i<stats->n_comm; ++i)
      predicted_model->collection[predicted_model->start_stats].commData[i] = stats->commData[i];
    ++predicted_model->cur_stats;
    ++predicted_model->start_stats;

  } else {

    if (predicted_model->parameters == NULL) {     // time to create the new prediction model
      // allocate parameters
      predicted_model->model_valid = new bool[stats->n_objs];
      predicted_model->parameters = new double*[stats->n_objs];
      for (i=0; i<stats->n_objs; ++i) predicted_model->parameters[i] = new double[predicted_model->predictor->num_params];
      for (i=0; i<stats->n_objs; ++i) {
	// initialization
	predicted_model->predictor->initialize_params(predicted_model->parameters[i]);
	predicted_model->predictor->print(predicted_model->parameters[i]);

	model_done = Marquardt_solver(predicted_model, i);
	// always initialize to false for conservativity
	predicted_model->model_valid[i] = false;
	CkPrintf("LB: Model for object %d %s\n",i,model_done?"found":"not found");
	predicted_model->predictor->print(predicted_model->parameters[i]);
      }

      if (predicted_model->model_valid) {
	CkPrintf("LB: New model completely constructed\n");
      } else {
	CkPrintf("LB: Construction of new model failed\n");
      }

    } else {     // model already constructed, update it

      double *error_model = new double[stats->n_objs];
      double *error_default = new double[stats->n_objs];

      CkPrintf("Error in estimation:\n");
      for (i=0; i<stats->n_objs; ++i) {
	error_model[i] = stats->objData[i].wallTime-predicted_model->predictor->predict(predicted_model->collection[(predicted_model->start_stats-1)%predicted_model->n_stats].objData[i].wallTime,predicted_model->parameters[i]);
	error_default[i] = stats->objData[i].wallTime-predicted_model->collection[(predicted_model->start_stats-1)%predicted_model->n_stats].objData[i].wallTime;
	CkPrintf("object %d: real time=%f, model error=%f, default error=%f\n",i,stats->objData[i].wallTime,error_model[i],error_default[i]);
      }

      // save statistics in the last position
      if (predicted_model->start_stats >= predicted_model->n_stats) predicted_model->start_stats -= predicted_model->n_stats;
      if (predicted_model->cur_stats < predicted_model->n_stats) ++predicted_model->cur_stats;

      predicted_model->collection[predicted_model->start_stats].objData.resize(stats->n_objs);
      predicted_model->collection[predicted_model->start_stats].commData.resize(stats->n_comm);

      predicted_model->collection[predicted_model->start_stats].n_objs = stats->n_objs;
      predicted_model->collection[predicted_model->start_stats].n_migrateobjs = stats->n_migrateobjs;
      predicted_model->collection[predicted_model->start_stats].n_comm = stats->n_comm;
      for (i=0; i<stats->n_objs; ++i)
	predicted_model->collection[predicted_model->start_stats].objData[i] = stats->objData[i];
      for (i=0; i<stats->n_comm; ++i)
	predicted_model->collection[predicted_model->start_stats].commData[i] = stats->commData[i];      
      ++predicted_model->start_stats;      

      // check if model is ok
      // the check can be performed even if the model is not valid since it will
      // releave which objects are wrongly updated and will try to fix them

      // the update of the model is done if the model does not approximate
      // sufficiently well the underlining function or if the time-invariante
      // approach is performing better
      for (i=0; i<stats->n_objs; ++i) {
        //if (fabs(error_model[i]) > 0.2*stats->objData[i].wallTime || fabs(error_model[i]) > fabs(error_default[i])) {
        if (fabs(error_model[i]) > fabs(error_default[i])) {  // no absolute error check
	  predicted_model->model_valid[i] = false;
	  // model wrong, rebuild it now
	  predicted_model->predictor->initialize_params(predicted_model->parameters[i]);
	  model_done = Marquardt_solver(predicted_model, i);
	  CkPrintf("LB: Updated model for object %d %s",i,model_done?"success":"failed. ");
	  predicted_model->predictor->print(predicted_model->parameters[i]);
	}
	if (fabs(error_model[i]) < fabs(error_default[i])) predicted_model->model_valid[i] = true;
      }

    }

    // use the model to update statistics
    double *param;
    for (int i=0; i<stats->n_objs; ++i) {
      if (predicted_model->model_valid[i]) {
	param = predicted_model->parameters[i];
	stats->objData[i].wallTime = predicted_model->predictor->predict(stats->objData[i].wallTime, param);
#if CMK_LB_CPUTIMER
	stats->objData[i].cpuTime = predicted_model->predictor->predict(stats->objData[i].cpuTime, param);
#endif
      }
    }

  }

#endif
}

/*@}*/
