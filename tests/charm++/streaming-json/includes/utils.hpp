#include <random>
#include <string>
#include <vector>
#include <json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;
using namespace std;

vector<string> load_names_from_file(const string& filename) {
    vector<string> names;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    
    string name; 
    while (getline(file, name)) {
        if (!name.empty()) {
            names.push_back(name);
        }
    }
    
    file.close();
    return names;
}

// Generate a random int from [min, max)
int random_int(int min, int max) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// Generate a random float from [min, max)
double random_double(double min, double max) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

// Generate a random name
string random_name(const vector<string>& first_names, const vector<string>& last_names) {
    return first_names[random_int(0, first_names.size() - 1)] + " " + last_names[random_int(0, last_names.size() - 1)];
}

// Function to generate a single biometric record
json generate_biometric_record(vector<string> first_names, vector<string> last_names) {
    json record;
    record["name"] = random_name(first_names, last_names);
    record["age"] = random_int(18, 70);
    
    json biometrics;
    biometrics["heart_rate"] = random_int(50, 120);
    biometrics["steps"] = random_int(1000, 15000); 
    biometrics["weight_kg"] = random_double(50.0, 100.0);
    biometrics["height_m"] = random_double(1.5, 2.0);
    
    record["biometrics"] = biometrics;
    
    return record;
}


// Generates sample json data.
// Expects /data/first_names.txt and /data/last_name.txt to exist 
json generate_and_save_json(int n, const string& filename) {
    vector<string> first_names = load_names_from_file("data/first_names.txt");
    vector<string> last_names = load_names_from_file("data/last_names.txt");

    json records = json::array();
    
    for (int i = 0; i < n; i++) {
        records.push_back(generate_biometric_record(first_names, last_names));
    }
    
    // Save to file
    ofstream file(filename);
    if (file.is_open()) {
        file << records.dump(4);
        file.close();
        cout << "Generated " << n << " biometric records and saved to " << filename << endl;
    } else {
        cerr << "Error opening file for writing!" << endl;
    }
    return records;
}

