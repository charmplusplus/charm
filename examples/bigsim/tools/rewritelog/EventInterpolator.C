
#include <EventInterpolator.h>

using namespace std;

int EventInterpolator::numCoefficients(string funcname){
  if(funcname == string("calc_pair_energy") )
	return 7;
  else if(funcname == string("calc_self_energy") )
    return 6;
  else if(funcname == string("angle"))
	return 3;
  else if(funcname == string("diherals"))
    return 3;
  else if(funcname == string("testcase"))
    return 2;
  else {
    throw new runtime_error("numCoefficients() does not know about some function name");
  }
}


EventInterpolator::EventInterpolator(char *table_filename){
  cout << "Loading timings file: " << table_filename << endl;
  ifstream paramtable(table_filename);

  // First pass, scan through file to count how many samples there are for each function
  while(paramtable.good()){
	string line_s;
	getline(paramtable,line_s);
	istringstream line(line_s);

	if(paramtable.good()){
	  string t1, t2, t3, t4, t5;
	  string funcname;

	  line >> t1 >> t2 >> t3 >> t4 >> t5;
	  line >> funcname;

	  sample_count[funcname]++;
	}
  }


  // Create a gsl interpolator workspace for each event/function
  for(map<string,unsigned long>::iterator i=sample_count.begin(); i!=sample_count.end();++i){
	string name = (*i).first;
	unsigned long samples = (*i).second;
	cout << "     > " << name << " has " << samples << " sampled timings" << endl;

    if(samples < numCoefficients(name) ){
        cerr << "FATAL ERROR: Not enough input timing samples for " << name << " which has " << numCoefficients(name) << " coefficients" << endl;
        throw new runtime_error("samples < numCoefficients");
    }
    else {
        work[name] = gsl_multifit_linear_alloc(samples,numCoefficients(name));
        X[name] = gsl_matrix_alloc (samples,numCoefficients(name));
        y[name] = gsl_vector_alloc (samples);
        c[name] = gsl_vector_alloc(numCoefficients(name));
        cov[name] = gsl_matrix_alloc(numCoefficients(name),numCoefficients(name));

        for(int i=0;i<numCoefficients(name);++i){
            gsl_vector_set(c[name],i,1.0);
        }

    }
  }

//  paramtable.seekg(0,ios_base::beg); // rewind
  paramtable.close(); // I do this because seeking was failing earlier. why???

  ifstream paramtable2(table_filename);

    cout << "here 2" << endl;

  // Second Pass, scan through the file to load
  while(paramtable2.good()){
	string line_s;
	getline(paramtable2,line_s);
	istringstream line(line_s);

	if(paramtable2.good()){
        string t1, t2, t3, t4, t5;
        string funcname;

        line >> t1 >> t2 >> t3 >> t4 >> t5;
        line >> funcname;

        if(t4 == string("TRACEBIGSIM")){
            unsigned i = Xcount[funcname] ++;
            gsl_matrix * x = X[funcname];
            if(funcname == string("calc_pair_energy")){
                double d1,d2,d3,d4,d5,d6,d7,d8,d9, t1;
                unsigned i1,i2,i3,i4,i5,i6;
                string s1;
                line >> s1 >> d1 >> d2 >> i1 >> i2 >> i3 >> i4 >> i5 >>
                        i6 >> d3 >> d4 >> d5 >> d6 >> d7 >> d8 >> d9 >> t1;
                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, min(d1,d2) );
                gsl_matrix_set(x,i,2, 1.0/( (i1-i4)*(i1-i4)+(i2-i5)*(i2-i5)+(i3-i6)*(i3-i6) ) );
                // Ignore d3, d8,d9
                gsl_matrix_set(x,i,3, d4 );
                gsl_matrix_set(x,i,4, d5 );
                gsl_matrix_set(x,i,5, d6 );
                gsl_matrix_set(x,i,6, d7 );
                gsl_vector_set(y[funcname],i,t1);
            }
            else if(funcname == string("angle")){
                double d1, d2, t1, t2;
                line >> d1 >> d2 >> t1 >> t2;
                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, d1 );
                gsl_matrix_set(x,i,2, d2 );
                gsl_vector_set(y[funcname],i,t2-t1);
            }
            else if(funcname == string("testcase")){
                double d1, d2, t1;
                line >> d1 >> d2 >> t1;
                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, d1*d2 );
                gsl_vector_set(y[funcname],i,t1);
            }
        }
	}
  }


    // Perform a sanity check now

    for(map<string, gsl_multifit_linear_workspace *>::iterator i=work.begin();i!=work.end();++i){
        assert(sample_count[(*i).first]==Xcount[(*i).first]);
    }

    cout << "Performing Least Squared Fits to sampled time data" << endl;

    //  Now do Least Square Fit: Find C where y=Xc
    map<string, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        string name = (*i).first;
        assert(! gsl_multifit_linear(X[name],y[name],c[name],cov[name],&chisqr[name],work[name]));
        gsl_matrix_free(X[name]);
        gsl_vector_free(y[name]);
        cout << "     > " << name << " has chisqr=" << chisqr[name] << endl;
    }


}


double EventInterpolator::predictTime(string name, double *params) {

    // Estimate time for a given set of parameters p
    gsl_vector *desired_params;

    desired_params = gsl_vector_alloc(numCoefficients(name));

    if(name == string("testcase")){
        gsl_vector_set(desired_params,0,1.0);
        gsl_vector_set(desired_params,1,params[0]*params[1]);
    } else {
        cerr << "FATAL ERROR: predictTime does not understand " << name << " yet" << endl;
        throw new runtime_error("predictTime not yet done");
    }

    double desired_time, desired_time_err;
    assert(! gsl_multifit_linear_est(desired_params,c[name],cov[name],&desired_time,&desired_time_err));

    // We now have a predicted time for the desired parameters

    gsl_vector_free(desired_params);
    return desired_time;

}



/** Free the gsl arrays and workspaces */
EventInterpolator::~EventInterpolator(){
    map<string, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        string name = (*i).first;
        gsl_multifit_linear_free(work[name]);
        gsl_matrix_free(cov[name]);
        gsl_vector_free(c[name]);
    }
}




