
#include <EventInterpolator.h>


int EventInterpolator::numCoefficients(std::string funcname){
  if(funcname == std::string("calc_pair_energy") )
	return 9;
  if(funcname == std::string("calc_self_energy") )
	return 9;
  else if(funcname == std::string("angle"))
	return 2;
  else if(funcname == std::string("diherals"))
	return 2;
}





EventInterpolator::EventInterpolator(char *table_filename){
  
  work = NULL;
  
  std::ifstream paramtable(table_filename);

  // First pass, scan through file to count how many samples there are for each function
  while(paramtable.good()){
	std::string line_s;
	getline(paramtable,line_s);
	std::istringstream line(line_s);
	
	if(paramtable.good()){
	  std::string t1, t2, t3, t4, t5;
	  std::string funcname;
	  
	  line >> t1 >> t2 >> t3 >> t4 >> t5;
	  line >> funcname;

	  sample_count[funcname]++;
	}
  }


  // Create a gsl interpolator workspace for each event/function
  for(std::map<std::string,unsigned long>::iterator i=sample_count.begin(); i!=sample_count.end();++i){
	std::string name = (*i).first;
	unsigned long occur = (*i).second;
	std::cout << "function " << name << " has " << occur << " occurrences" << std::endl;
	
	work[(*i).first] = gsl_multifit_linear_alloc(occur,numCoefficients(name));
  }

  paramtable.close();

  // Second Pass, scan through the file to load 
  while(paramtable.good()){
	std::string line_s;
	getline(paramtable,line_s);
	std::istringstream line(line_s);
	
	if(paramtable.good()){
	  std::string t1, t2, t3, t4, t5;
	  std::string funcname;
	  
	  line >> t1 >> t2 >> t3 >> t4 >> t5;
	  line >> funcname;
	  
	  
	  
	  
	}
  }


  //  Find C     where y=Xc

  gsl_matrix *X;  // Each row is a set of parameters  [1, a, a^2, b, b^2, a*b] for each input parameter set
  gsl_vector *y;  // vector of cycle accurate times for each input parameter set


  X = gsl_matrix_alloc (n,cs);
  y = gsl_vector_alloc (n);
  c = gsl_vector_alloc(cs);
  cov = gsl_matrix_alloc(cs,cs);

  // Load parameters and times from input file
  for(int i=0;i<n;i++){
      double temp1, temp2;
      paramtable >> temp1;
      paramtable >> temp2;
      
      gsl_matrix_set(X,i,0,1.0);
      gsl_matrix_set(X,i,1,temp1);
      gsl_matrix_set(X,i,2,temp1*temp1);
      gsl_matrix_set(X,i,3,temp2);
      gsl_matrix_set(X,i,4,temp2*temp2);
      gsl_matrix_set(X,i,5,temp1*temp2);
      
      double temp;
      paramtable >> temp;
      gsl_vector_set(y,i,temp);
  }

  // Do we need to initialize c
  for(int j=0;j<cs;j++){
      gsl_vector_set(c,j,1.0);
  }

  gsl_multifit_linear(X,y,c,cov,&chisqr,work);

  gsl_matrix_free(X);
  gsl_vector_free(y);

}


double EventInterpolator::predictTime(double *params) {

  // Estimate time for a given set of parameters p
  gsl_vector *desired_params;
  
  desired_params = gsl_vector_alloc(cs);

  gsl_vector_set(desired_params,0,1.0);
  gsl_vector_set(desired_params,1,params[0]);
  gsl_vector_set(desired_params,2,params[0]*params[0]);
  gsl_vector_set(desired_params,3,params[1]);
  gsl_vector_set(desired_params,4,params[1]*params[1]);
  gsl_vector_set(desired_params,5,params[0]*params[1]);
  
  
  double desired_time, desired_time_err;
  gsl_multifit_linear_est(desired_params,c,cov,&desired_time,&desired_time_err);

  // We now have a predicted time for the desired parameters

    gsl_vector_free(desired_params);
  return desired_time;
  
}

EventInterpolator::~EventInterpolator(){
  gsl_multifit_linear_free(work);
  gsl_matrix_free(cov);
  gsl_vector_free(c);
}




