#include <iostream>
#include <vector>
#include <cstdlib>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "CImg.h"
#include <unsupported/Eigen/SparseExtra>
#include <experimental/filesystem>
#include <boost/algorithm/string.hpp>
#include "omp.h"
#include <thread>
#include <mutex>


using namespace std;
using namespace cimg_library;
using namespace Eigen;
namespace fs = std::experimental::filesystem;

mutex mu;
typedef Triplet<double> T;

vector<string> paths;

int print(CImg<float> &I, int c)
{
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			cout << I(j, i, 0, c) << " ";
		}
		cout << '\n';
	}
	return 0;
}


void decompose(int begin, int end)
{
	vector<string> split_list;
	string image_name, input_image_path;
	float log_val, max_val, luma_val, wt_alb_reg, max_diff = 0, val, sum_val = 0.0, wA, wB;

	//#pragma omp parallel for
	for(int id = begin; id < end; id++){
		input_image_path = paths[id];
		boost::split(split_list, input_image_path, boost::is_any_of("/"));
		image_name = split_list[7];
		CImg<float> image(input_image_path.c_str());

		
		image = image/255;
		image.resize(112, 112, 1, 3, 5, 0, 0, 0, 0, 0);	

		CImg<float> luma    = image.get_channel(0) + image.get_channel(1) + image.get_channel(2);
		CImg<float> luma_1c = luma;
		CImg<float> chroma  = image;
		luma.append(luma_1c, 'c', 1);
		luma.append(luma_1c, 'c', 1);
		luma += 1e-6;
		chroma.div(luma);
		luma_1c /= 3;
		luma /= 3;
		/*cout << "Image - 1" << '\n';
		print(image, 0);
		cout << "Image - 2" << '\n';
		print(image, 1);
		cout << "Image - 3" << '\n';
		print(image, 2);
		cout << '\n';
		cout << "Luma - 1" << '\n';
		print(luma, 0);
		cout << "Luma - 2" << '\n';
		print(luma, 1);
		cout << "Luma - 3" << '\n';
		print(luma, 2);
		cout << '\n';
		cout << "Chroma - 1" << '\n';
		print(chroma, 0);
		cout << "Chroma - 2" << '\n';
		print(chroma, 1);
		cout << "Chroma - 3" << '\n';
		print(chroma, 2);*/
		int w = image.width(), h = image.height(), e, f, e1, f1, n = w*h, reg_nn = 2, rand_num, k = 0, min, max, reg_size, rows, cols;
		vector<vector<int> > edges(n*reg_nn, vector<int> (2));
		for(int i = 0; i < n; i++){
			for(int j = 0; j < reg_nn; j++){
				rand_num = rand()%n;
				while(rand_num == i) rand_num = rand()%n;
				min = (i>rand_num)?rand_num:i;
				max = (i>rand_num)?i:rand_num;
				edges[k][0] = min;
				edges[k][1] = max;
				k++;
			}
		}
		sort(edges.begin(), edges.end());
		edges.erase(unique(edges.begin(), edges.end()), edges.end());

		reg_size = edges.size();
		//cout << "Reg size: " << reg_size << '\n';

		rows = 3*n+3*reg_size;
		cols = 4*n;

		vector<float> diff(reg_size);
		VectorXd x(cols), b(rows);
		b.setZero();
		vector<T> tripletList;
		tripletList.reserve(2*rows);
		tripletList.clear();
		for(int i = 0; i < n; i++){
			e = i/h;
			f = i%h;
			luma_val = luma_1c(e, f, 0, 0)*0.9999 + 0.0001 + 0.1;
			tripletList.push_back(T(i, i, luma_val));
			tripletList.push_back(T(i, i+3*n, luma_val));
			tripletList.push_back(T(i+n, i+n, luma_val));
			tripletList.push_back(T(i+n, i+3*n, luma_val));
			tripletList.push_back(T(i+2*n, i+2*n, luma_val));
			tripletList.push_back(T(i+2*n, i+3*n, luma_val));

			log_val  = log(image(e, f, 0, 0) + 0.001);
			if(log_val < -6) log_val = -6;
			b[i]     = log_val*luma_val;

			log_val  = log(image(e, f, 0, 1) + 0.001);
			if(log_val < -6) log_val = -6;
			b[i+n]   = log_val*luma_val;

			log_val  = log(image(e, f, 0, 2) + 0.001);
			if(log_val < -6) log_val = -6;
			b[i+2*n] = log_val*luma_val;
		}

		for(int i = 0; i < reg_size; i++){
			e  = edges[i][0]/h;
			f  = edges[i][0]%h;
			e1 = edges[i][1]/h;
			f1 = edges[i][1]%h;
			diff[i] = abs(chroma(e, f, 0, 0) - chroma(e1, f1, 0, 0)) + abs(chroma(e, f, 0, 1) - chroma(e1, f1, 0, 1)) + abs(chroma(e, f, 0, 2) - chroma(e1, f1, 0, 2));
			if(diff[i] > max_diff) max_diff = diff[i];
		}

		for(int i = 0; i < reg_size; i++){
			e  = edges[i][0]/h;
			f  = edges[i][0]%h;
			e1 = edges[i][1]/h;
			f1 = edges[i][1]%h;
			val = (1-diff[i]/max_diff)*(sqrt(luma_1c(e, f, 0, 0)*luma_1c(e1, f1, 0, 0)));
			tripletList.push_back(T(3*n+i, edges[i][0], 0.1*val));
			tripletList.push_back(T(3*n+i, edges[i][1], -0.1*val));
			tripletList.push_back(T(3*n+reg_size+i, n+edges[i][0], 0.1*val));
			tripletList.push_back(T(3*n+reg_size+i, n+edges[i][1], -0.1*val));
			tripletList.push_back(T(3*n+2*reg_size+i, 2*n+edges[i][0], 0.1*val));
			tripletList.push_back(T(3*n+2*reg_size+i, 2*n+edges[i][1], -0.1*val));
		}

		SparseMatrix<double> A(rows, cols);
		A.setFromTriplets(tripletList.begin(), tripletList.end());
		/*cout << "A: " << '\n'; 
		for(int i = 0; i < 5; i++){
			for(int j = 0; j < 5; j++) cout << A.coeffRef(i, j) << " ";
			cout << '\n';
		}

		cout << "b: " << '\n';
		for(int i = 0; i < 5; i++) cout << b[i] << '\n';*/
		

		LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
		lscg.setMaxIterations(200);
		lscg.setTolerance(pow(10, -5));
		lscg.compute(A);
		x = lscg.solve(b);

		//for(int i = 0; i < 5; i++) cout << x[i] << " ";
		//cout << '\n';

		CImg<float> albedo(w, h, 1, 3);
		CImg<float> shading(w, h, 1, 1);

		for(int i = 0; i < n; i++){
			e = i/h;
			f = i%h;
			albedo(e, f, 0, 0) = exp(x[i]);
			albedo(e, f, 0, 1) = exp(x[i+n]);
			albedo(e, f, 0, 2) = exp(x[i+2*n]);
			sum_val += exp(x[i]) + exp(x[i+n]) + exp(x[i+2*n]);
		}

		wA = sum_val/(3*n);
		sum_val = 0;

		for(int i = 0; i < n; i++){
			e = i/h;
			f = i%h;
			shading(e, f, 0, 0) = exp(x[i+3*n]);
			sum_val += exp(x[i+3*n]);
		}
		wB = sum_val/n;
		albedo  = (albedo/wA)*sqrt(wA*wB);
		shading = (shading/wB)*sqrt(wA*wB);
		
		albedo.normalize(0,255);
		shading.normalize(0,255);
		//albedos[i] = albedo;
		//shadings[i] = shading;
		mu.lock();
		albedo.save(("/home/sumanth/imagenet/ILSVRC/albedo_112/val/"+image_name).c_str());
		shading.save(("/home/sumanth/imagenet/ILSVRC/shading_112/val/"+image_name).c_str()); 
		mu.unlock();
		//image.save_jpeg(("/home/sumanth/imagenet/ILSVRC/" + image_name).c_str()); 
	}
}

int main()
{
	string path = "/home/sumanth/imagenet/ILSVRC/images_224/val/", input_image_path, image_name;
	int i = 0, max_threads = 30, block_size;
	//vector<string> split_list;
	//string image_name, input_image_path;
	for(const auto & entry : fs::directory_iterator(path)){paths.push_back(entry.path());}
		//if(i == 9) break;
		//i++;
	/*while(i < paths.size()){
		decompose(++i);
		/*thread first(decompose, ++i);
		thread second(decompose, ++i);
		thread third(decompose, ++i);
		thread fourth(decompose, ++i);
		thread fifth(decompose, ++i);
		thread sixth(decompose, ++i);
		thread seventh(decompose, ++i);
		thread eighth(decompose, ++i);
		thread ninth(decompose, ++i);
		thread tenth(decompose, ++i);
		thread first_(decompose, ++i);
		thread second_(decompose, ++i);
		thread third_(decompose, ++i);
		thread fourth_(decompose, ++i);
		thread fifth_(decompose, ++i);
		thread sixth_(decompose, ++i);
		thread seventh_(decompose, ++i);
		thread eighth_(decompose, ++i);
		thread ninth_(decompose, ++i);
		thread tenth_(decompose, ++i);
		thread first__(decompose, ++i);
		thread second__(decompose, ++i);
		thread third__(decompose, ++i);
		thread fourth__(decompose, ++i);
		thread fifth__(decompose, ++i);
		thread sixth__(decompose, ++i);
		thread seventh__(decompose, ++i);
		thread eighth__(decompose, ++i);
		thread ninth__(decompose, ++i);
		thread tenth__(decompose, ++i);*/
		//thread first___(decompose, 30*block_size, paths.size());
		
		/*first.join();
		second.join();
		third.join();
		fourth.join();
		fifth.join();
		sixth.join();
		seventh.join();
		eighth.join();
		ninth.join();
		tenth.join();
		first_.join();
		second_.join();
		third_.join();
		fourth_.join();
		fifth_.join();
		sixth_.join();
		seventh_.join();
		eighth_.join();
		ninth_.join();
		tenth_.join();
		first__.join();
		second__.join();
		third__.join();
		fourth__.join();
		fifth__.join();
		sixth__.join();
		seventh__.join();
		eighth__.join();
		ninth__.join();
		tenth__.join();
	}*/
	block_size = paths.size()/max_threads;

	//cout << paths.size() << " " << block_size;
	
	//decompose(0, 1);
	thread first(decompose, 0, block_size);
	thread second(decompose, block_size, 2*block_size);
	thread third(decompose, 2*block_size, 3*block_size);
	thread fourth(decompose, 3*block_size, 4*block_size);
	thread fifth(decompose, 4*block_size, 5*block_size);
	thread sixth(decompose, 5*block_size, 6*block_size);
	thread seventh(decompose, 6*block_size, 7*block_size);
	thread eighth(decompose, 7*block_size, 8*block_size);
	thread ninth(decompose, 8*block_size, 9*block_size);
	thread tenth(decompose, 9*block_size, 10*block_size);
	thread first_(decompose, 10*block_size, 11*block_size);
	thread second_(decompose, 11*block_size, 12*block_size);
	thread third_(decompose, 12*block_size, 13*block_size);
	thread fourth_(decompose, 13*block_size, 14*block_size);
	thread fifth_(decompose, 14*block_size, 15*block_size);
	thread sixth_(decompose, 15*block_size, 16*block_size);
	thread seventh_(decompose, 16*block_size, 17*block_size);
	thread eighth_(decompose, 17*block_size, 18*block_size);
	thread ninth_(decompose, 18*block_size, 19*block_size);
	thread tenth_(decompose, 19*block_size, 20*block_size);
	thread first__(decompose, 20*block_size, 21*block_size);
	thread second__(decompose, 21*block_size, 22*block_size);
	thread third__(decompose, 22*block_size, 23*block_size);
	thread fourth__(decompose, 23*block_size, 24*block_size);
	thread fifth__(decompose, 24*block_size, 25*block_size);
	thread sixth__(decompose, 25*block_size, 26*block_size);
	thread seventh__(decompose, 26*block_size, 27*block_size);
	thread eighth__(decompose, 27*block_size, 28*block_size);
	thread ninth__(decompose, 28*block_size, 29*block_size);
	thread tenth__(decompose, 29*block_size, paths.size());
	//thread first___(decompose, 30*block_size, paths.size());
	
	first.join();
	second.join();
	third.join();
	fourth.join();
	fifth.join();
	sixth.join();
	seventh.join();
	eighth.join();
	ninth.join();
	tenth.join();
	first_.join();
	second_.join();
	third_.join();
	fourth_.join();
	fifth_.join();
	sixth_.join();
	seventh_.join();
	eighth_.join();
	ninth_.join();
	tenth_.join();
	first__.join();
	second__.join();
	third__.join();
	fourth__.join();
	fifth__.join();
	sixth__.join();
	seventh__.join();
	eighth__.join();
	ninth__.join();
	tenth__.join();


	/*CImg<float> albedo(112, 112, 1, 3);
	CImg<float> shading(112, 112, 1, 1);

	for(int i = 0; i < paths.size(); i++){
		input_image_path = paths[i];
		albedo = albedos[i];
		shading = shadings[i];
		boost::split(split_list, input_image_path, boost::is_any_of("/"));
		image_name = split_list[7];
		albedo.save_jpeg();

	}*/

	/*first.detach();
	second.detach();
	third.detach();
	fourth.detach();
	fifth.detach();
	sixth.detach();
	seventh.detach();
	eighth.detach();
	ninth.detach();
	tenth.detach();
	first_.detach();
	second_.detach();
	third_.detach();
	fourth_.detach();
	fifth_.detach();
	sixth_.detach();
	seventh_.detach();
	eighth_.detach();
	ninth_.detach();
	tenth_.detach();
	first__.detach();
	second__.detach();
	third__.detach();
	fourth__.detach();
	fifth__.detach();
	sixth__.detach();
	seventh__.detach();
	eighth__.detach();
	ninth__.detach();
	tenth__.detach();*/
	//first___.join();
    /*for(const auto & entry : fs::directory_iterator(path)){
    	input_image_path = entry.path();
    	thread first(decompose, input_image_path);
    	thread second(decompose, input_image_path);
    	thread third(decompose, input_image_path);
    	thread fourth(decompose, input_image_path);
    	first.join();
    	second.join();
    	third.join();
    	fourth.join();
    	//decompose(input_image_path);
    	if(i == 0) break;
    	i++;
    }*/
    //cout << paths.size() << '\n';

    return 0;
}