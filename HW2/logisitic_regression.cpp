#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock  
#include <pthread.h>
#include <unistd.h>
#include <pthread.h>
#define NUM_THREADS 3
#define M 2177020
#define N 11392
using namespace std;

double sigmoid(double x){
	return 1.0 / (1 + exp(-x));
}

struct thread_data
{
	int start;
	int end;
	int thread_id;
	double *result;
};


vector< vector<int> > data_matrix;
vector<double> weights(N);


void *calcCostThread(void *threadarg)
{
   struct thread_data *my_data;

   my_data = (struct thread_data *) threadarg;

   
   double total_sum = 0;
   int s = (my_data->start);
   int e = (my_data->end);
   for (int i = s; i < e; ++i)
   {
   		double temp_sum = 0;
   		for (int k = 1; k < data_matrix[i].size(); ++k)
		{
			temp_sum += weights[ data_matrix[i][k] ];
		}
		double h = sigmoid(temp_sum);
		double error = (h - data_matrix[i][0]);
		double cost = pow((h-error),2)/2.0;
		total_sum += cost;
   }
   //cout<<"total_sum: "<<total_sum<<endl;
   *(my_data->result) = total_sum;
   pthread_exit(NULL);
}

vector<double> stochGradAscent(int intNum){
	int m = M;
	int n = 11392;
	for (int i = 0; i < intNum; ++i)
	{
		cout<<"The "<<i<<" is dealing"<<endl;

		std::random_shuffle(data_matrix.begin(), data_matrix.end());
		
		for (int j = 0; j < m; ++j)
		{
			double alpha = 4 / (1.0 + j + i) + 0.01;
			double sum = 0;
			for (int k = 1; k < data_matrix[j].size(); ++k)
			{
				sum += weights[ data_matrix[j][k] ];
			}

			double h = sigmoid(sum);
			double error = data_matrix[j][0] - h;

			double mul = alpha * error;
			for (int k = 1; k < data_matrix[j].size(); ++k)
			{
				weights[ data_matrix[j][k] ] += mul;
			}
		}

		int s[3] = {0, m/3, 2*m/3};
		int e[3] = {m/3, 2*m/3, m};
		double res_mid[3] = {0, 0, 0};
		int rc;
		pthread_t threads[NUM_THREADS];
		pthread_attr_t attr;
		
		thread_data td[NUM_THREADS];
		void *status;

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		for (int t = 0; t < NUM_THREADS; ++t)
		{
			
			td[t].thread_id = t;
			td[t].start = s[t];
			td[t].end = e[t];
			td[t].result = &res_mid[t];
			
			rc = pthread_create(&threads[t], NULL, calcCostThread, (void *)&td[t]);
			
			if (rc){
         		cout << "Error:unable to create thread," << rc << endl;
         		exit(-1);
      		}
		}

		pthread_attr_destroy(&attr);
		
		for (int t = 0; t < NUM_THREADS; ++t)
		{
			
			rc = pthread_join(threads[t], &status);
			if (rc){
         		cout << "Error:unable to join," << rc << endl;
         		exit(-1);
      		}
      		
		}
		
		double tot = 0;
		for (int t = 0; t < NUM_THREADS; ++t)
		{
			tot += res_mid[t];
		}
		cout<<"J(theta) = "<<tot/m<<endl;
	}
	return weights;
}




int main(){
	
	ifstream readfile;
	readfile.open("/Users/All4win/Documents/Three/DataMining/HW2/train_data.txt",ios::in);
	string line;
	int num;
	char s;
	int count = M;
	while(getline(readfile,line) &&count--)
    {
        istringstream ss(line);

        vector<int> temp_v;
        ss>>num;
        temp_v.push_back(num);
        while(ss>>s>>num){
        	temp_v.push_back(num);
        }
        data_matrix.push_back(temp_v);
    }
    readfile.close();
    cout<<"read is over"<<endl;

    vector<double> result = stochGradAscent(100);
    // vector<double> result(10);
    ofstream writefile;
    ostringstream os;
    writefile.open("/Users/All4win/Documents/Three/DataMining/HW2/cpp_weight.txt", ios::out);
    os << result[0];
    for (int i = 1; i < result.size(); ++i)
    {
    	os <<", "<< result[i];
    }
    string res = os.str();
    writefile << res;
    writefile.close();
	return 0;
}