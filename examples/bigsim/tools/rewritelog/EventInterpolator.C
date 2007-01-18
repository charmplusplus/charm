
#include <EventInterpolator.h>

using namespace std;

int EventInterpolator::numCoefficients(string funcname){
    if( funcname == string("calc_self_energy_merge_fullelect") ||
        funcname == string("calc_pair_energy_merge_fullelect") ||
        funcname == string("calc_self_merge_fullelect") ||
        funcname == string("calc_pair_merge_fullelect") ||
        funcname == string("calc_self") ||
        funcname == string("calc_pair") ||
        funcname == string("calc_self_energy") ||
        funcname == string("calc_pair_energy") ) {
            return 3;
        }
    else if(funcname == string("angle") || funcname == string("dihedrals")){
            return 3;
        }
    else if(funcname == string("*integrate*") ){
            return 2;
        }
    else {
        cerr << "Unknown function: \"" << funcname << "\"" << endl;
        throw new runtime_error("numCoefficients() does not know about some function name");
    }
}


EventInterpolator::EventInterpolator(char *table_filename){
    cout << "Loading timings file: " << table_filename << endl;
    ifstream accurateTimeTable(table_filename);

    // First pass, scan through file containing cycle accurate times to count
    // how many samples there are for each function
    while(accurateTimeTable.good()){
        string line_s;
        getline(accurateTimeTable,line_s);
        istringstream line(line_s);

        string temp("");
        while(temp != string("TRACEBIGSIM") && line.good() && accurateTimeTable.good() ){
            line >> temp;
//             cout << "G:  " << temp << endl;
        }
        line >> temp; // gobble up one more worthless bit of input line
//         cout << "G2:  " << temp << endl;

        if(line.good() && accurateTimeTable.good()){
            string funcname;
            line >> funcname;
//             cout << "F=" << funcname << endl;
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

    //  accurateTimeTable.seekg(0,ios_base::beg); // rewind
    accurateTimeTable.close(); // I do this because seeking was failing earlier. why???

    ifstream accurateTimeTable2(table_filename);

//     cout << "HERE" << endl;

    // Second Pass, scan through the file to load
    while(accurateTimeTable2.good()){
        string line_s;
        getline(accurateTimeTable2,line_s);
        istringstream line(line_s);

        string temp("");
        while(temp != string("TRACEBIGSIM") && line.good() && accurateTimeTable2.good() ){
            line >> temp;
//             cout << "G:  " << temp << endl;
        }
        line >> temp; // gobble up one more worthless bit of input line
//         cout << "G2:  " << temp << endl;

        if(line.good() && accurateTimeTable2.good()){
            string funcname;
            line >> funcname;

//             cout << "funcname=" << funcname << endl;

            unsigned i = Xcount[funcname] ++;
            gsl_matrix * x = X[funcname];
            if( funcname == string("calc_self_energy_merge_fullelect") ||
                funcname == string("calc_pair_energy_merge_fullelect") ||
                funcname == string("calc_self_merge_fullelect") ||
                funcname == string("calc_pair_merge_fullelect") ||
                funcname == string("calc_self") ||
                funcname == string("calc_pair") ||
                funcname == string("calc_self_energy") ||
                funcname == string("calc_pair_energy") ){
                double d1,d2,d3,d4,d5,d6,d7,d8,d9, t1;
                unsigned i1,i2,i3,i4,i5,i6;

                line >> d1 >> d2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >>
                        d3 >> d4 >> d5 >> d6 >> d7 >> d8 >> t1;

                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, min(d1,d2) );
                gsl_matrix_set(x,i,2, d3 );

                gsl_vector_set(y[funcname],i,t1);
            }
            else if(funcname == string("angle") || funcname == string("testcase")){
                double d1, d2, t1;
                line >> d1 >> d2 >> t1;

                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, d1 );
                gsl_matrix_set(x,i,2, d2 );

                gsl_vector_set(y[funcname],i,t1);
            }
            else if(funcname == string("*integrate*")){
                double d1, d2, d3, d4, d5, d6, d7, t1;
                line >> d1 >> d2 >> d3 >> d4 >> d5 >> d6 >> d7 >> t1;

                gsl_matrix_set(x,i,0, 1.0);
                gsl_matrix_set(x,i,1, d2 );

                gsl_vector_set(y[funcname],i,t1);
            }

        }
    }


    // Perform a sanity check now

    for(map<string, gsl_multifit_linear_workspace *>::iterator i=work.begin();i!=work.end();++i){
        if(sample_count[(*i).first]!=Xcount[(*i).first]){
          cerr << "FATAL ERROR: sanity check failed: " << sample_count[(*i).first] << "!=" << Xcount[(*i).first] << "  :(" << endl;
       throw new runtime_error("sanity check failed");
        }
    }

    cout << "Performing Least Squared Fit to sampled time data" << endl;

    //  Now do Least Square Fit: Find C where y=Xc
    map<string, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        string name = (*i).first;
        assert(! gsl_multifit_linear(X[name],y[name],c[name],cov[name],&chisqr[name],work[name]));
        gsl_matrix_free(X[name]);
        gsl_vector_free(y[name]);
        cout << "     > " << name << " has chisqr=" << chisqr[name] << endl;
    }



    // Load in Parameter File which maps event id to function name and parameters
    cout << "Loading parameter files" << table_filename << endl;

    for(int i=0;i<102400;++i){
        char name[512];
        sprintf(name,"param.%d",i);
        ifstream parameterEventTable(name);

        if(parameterEventTable.good()){
            cout << "     >  Loading " << name << endl;

            while(parameterEventTable.good()){
                string line_s;
                getline(parameterEventTable,line_s);
                istringstream line(line_s);

                if(parameterEventTable.good()){
                    string t1, t2;
                    string funcname;
                    unsigned eventid;

                    line >> eventid;
                    line >> t1 >> t2;
                    line >> funcname;

                    if(t1 == string("TRACEBIGSIM")){
                        unsigned i = Xcount[funcname] ++;

                        if( funcname == string("calc_self_energy_merge_fullelect") ||
                            funcname == string("calc_pair_energy_merge_fullelect") ||
                            funcname == string("calc_self_merge_fullelect") ||
                            funcname == string("calc_pair_merge_fullelect") ||
                            funcname == string("calc_self_energy") ||
                            funcname == string("calc_pair_energy") ){
                            double d1,d2,d3,d4,d5,d6,d7,d8,d9, t1;
                            unsigned i1,i2,i3,i4,i5,i6;

                            line >> d1 >> d2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >>
                                    d3 >> d4 >> d5 >> d6 >> d7 >> d8 >> t1;

                            vector<double> params;
                            params.push_back( 1.0);
                            params.push_back( min(d1,d2) );
                            params.push_back( d3 );
                            eventparams[pair<unsigned,unsigned>(i,eventid)] = pair<string,vector<double> >(funcname,params);
                        }
                        else if(funcname == string("angle") || funcname == string("testcase")){
                            double d1, d2, t1;
                            line >> d1 >> d2 >> t1;

                            vector<double> params;
                            params.push_back( 1.0);
                            params.push_back( d1 );
                            params.push_back( d2 );
                            eventparams[pair<unsigned,unsigned>(i,eventid)] = pair<string,vector<double> >(funcname,params);

                        }
                        else if(funcname == string("*integrate*")){
                            double d1, d2, d3, d4, d5, d6, d7, t1;
                            line >> d1 >> d2 >> d3 >> d4 >> d5 >> d6 >> d7 >> t1;

                            vector<double> params;
                            params.push_back( 1.0);
                            params.push_back( d2 );
                            eventparams[pair<unsigned,unsigned>(i,eventid)] = pair<string,vector<double> >(funcname,params);

                        }
                    }
                }
            }

        }
        else{ // file was no good
            break;
        }

        parameterEventTable.close();
    }
}



double EventInterpolator::predictTime(const unsigned pe, const unsigned eventid) {
    return predictTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}


double EventInterpolator::predictTime(const pair<string,vector<double> > &p) {
    return predictTime(p.first,p.second);
}

double EventInterpolator::predictTime(const string &name, const vector<double> &params) {
    double val;
    double *p = new double[params.size()];
    for(int i=0;i<params.size();++i)
        p[i] = params[i];
    val = predictTime(name, p);
    delete[] p;
    return val;
}

double EventInterpolator::predictTime(const string &name, const double *params) {

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




