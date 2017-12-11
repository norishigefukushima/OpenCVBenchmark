#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xphoto.hpp>

#ifdef _DEBUG
#pragma comment (lib, "opencv_core320d.lib")
#pragma comment (lib, "opencv_highgui320d.lib")
#pragma comment (lib, "opencv_imgcodecs320d.lib")
#pragma comment (lib, "opencv_imgproc320d.lib")
#else
#pragma comment (lib, "opencv_core331.lib")
#pragma comment (lib, "opencv_highgui331.lib")
#pragma comment (lib, "opencv_imgcodecs331.lib")
#pragma comment (lib, "opencv_imgproc331.lib")
#pragma comment (lib, "opencv_photo331.lib")
#pragma comment (lib, "opencv_calib3d331.lib")
#pragma comment (lib, "opencv_objdetect331.lib")
#pragma comment (lib, "opencv_ximgproc331.lib")
#pragma comment (lib, "opencv_xphoto331.lib")

//#pragma comment (lib, "opencv_core320.lib")
//#pragma comment (lib, "opencv_highgui320.lib")
//#pragma comment (lib, "opencv_imgcodecs320.lib")
//#pragma comment (lib, "opencv_imgproc320.lib")
//#pragma comment (lib, "opencv_ximgproc320.lib")
//#pragma comment (lib, "opencv_xphoto320.lib")
#endif
using namespace cv;
using namespace std;

class CSV
{
	FILE* fp;
	bool isTop;
	long fileSize;
	std::string filename;
public:
	std::vector<double> argMin;
	std::vector<double> argMax;
	std::vector<std::vector<double>> data;
	std::vector<bool> filter;
	int width;
	void findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue);
	void initFilter();
	void filterClear();
	void makeFilter(int index, double val, double emax = 0.00000001);
	void readHeader();
	void readData();

	void init(std::string name, bool isWrite, bool isClear);
	CSV();
	CSV(std::string name, bool isWrite = true, bool isClear = true);

	~CSV();
	void write(std::string v);
	void write(double v);
	void endline();
};

void CSV::findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue)
{
	argMin.resize(data[0].size());
	argMax.resize(data[0].size());

	argMin[result_index] = DBL_MAX;
	argMax[result_index] = DBL_MIN;

	const int dsize = (int)argMin.size();
	if (isUseFilter)
	{
		for (int i = 0; i < data.size(); i++)
		{
			if (filter[i])
			{
				if (argMin[result_index] < data[i][result_index])
				{
					for (int j = 0; j < dsize; j++)
					{
						argMin[j] = data[i][j];
					}
				}
				if (argMax[result_index] > data[i][result_index])
				{
					for (int j = 0; j < dsize; j++)
					{
						argMax[j] = data[i][j];
					}
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < data.size(); i++)
		{
			if (argMin[result_index] < data[i][result_index])
			{
				for (int j = 0; j < dsize; j++)
				{
					argMin[j] = data[i][j];
				}
			}
			if (argMax[result_index] > data[i][result_index])
			{
				for (int j = 0; j < dsize; j++)
				{
					argMax[j] = data[i][j];
				}
			}
		}
	}
	minValue = argMin[result_index];
	maxValue = argMax[result_index];
}
void CSV::initFilter()
{
	filter.clear();
	for (int i = 0; i < data.size(); i++)
	{
		filter.push_back(true);
	}
}
void CSV::filterClear()
{
	for (int i = 0; i < data.size(); i++)
	{
		filter[i] = true;
	}
}
void CSV::makeFilter(int index, double val, double emax)
{
	for (int i = 0; i < data.size(); i++)
	{
		double diff = abs(data[i][index] - val);
		if (diff > emax)
		{
			filter[i] = false;
		}
	}
}
void CSV::readHeader()
{
	fseek(fp, 0, SEEK_END);
	fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	int countSep = 0;
	char str[1000];
	fgets(str, 1000, fp);
	for (int i = 0; i < strlen(str); i++)
	{
		switch (str[i])
		{
		case ',': countSep++;  break;
		}
	}
	width = countSep + 1;
}
void CSV::readData()
{
	char vv[100];
	char* str = new char[fileSize];
	fileSize = (long)fread(str, sizeof(char), fileSize, fp);

	int c = 0;
	vector<double> v;
	for (int i = 0; i < fileSize; i++)
	{
		if (str[i] == ',')
		{
			vv[c] = '\0';
			double d = atof(vv);
			c = 0;
			v.push_back(d);
		}
		else if (str[i] == '\n')
		{
			vv[c] = '\0';
			double d = atof(vv);
			c = 0;
			v.push_back(d);

			/*for(int n=0;n<v.size();n++)
			cout<<v[n]<<",";
			cout<<endl;*/

			data.push_back(v);
			v.clear();
		}
		else
		{
			vv[c] = str[i];
			c++;
		}
	}

	delete[] str;
}

void CSV::init(string name, bool isWrite, bool isClear)
{
	isTop = true;
	if (isWrite)
	{
		if (isClear)
		{
			fp = fopen(name.c_str(), "w");
		}
		else
		{
			fp = fopen(name.c_str(), "w+");
		}
		if (fp == NULL)
		{
			string n = name + "(1)";
			filename = n;
			fp = fopen(n.c_str(), "w");
		}
	}
	else
	{
		if (isClear)
		{
			fp = fopen(name.c_str(), "r");
		}
		else
		{
			fp = fopen(name.c_str(), "r+");
		}
		filename = name;
		if (fp == NULL)
		{
			cout << "file open error " << name << endl;
		}
		else
		{
			readHeader();
		}
	}
}
CSV::CSV()
{
	fp = NULL;
}
CSV::CSV(string name, bool isWrite, bool isClear)
{
	fp = NULL;
	init(name, isWrite, isClear);
}

CSV::~CSV()
{
	if (fp != NULL) fclose(fp);
	//ifstream file1(filename);
	//ofstream file2("backup"+filename);
	//char ch;
	//while(file1 && file1.get(ch))
	//{
	//	file2.put(ch);
	//}
}
void CSV::write(string v)
{
	if (isTop)
		fprintf(fp, "%s", v.c_str());
	else
		fprintf(fp, ",%s", v.c_str());

	isTop = false;
}
void CSV::write(double v)
{
	if (isTop)
		fprintf(fp, "%f", v);
	else
		fprintf(fp, ",%f", v);

	isTop = false;
}
void CSV::endline()
{
	fprintf(fp, "\n");
	isTop = true;
}

class Performance
{
public:
	vector<Point3d> time;
	vector<int> size;
	virtual void function(Mat& src, Mat& dest) = 0;
	enum
	{
		SIZE = -1,
		TIME_MEAN = 0,
		TIME_STD = 1,
		TIME_MEDIAN = 2,
		TIME_PIXEL_PER_MEDIAN = 3,
	};
	void csvout(CSV& csv, string name, int outputDataType = Performance::TIME_MEDIAN)
	{
		const int datasize = (int)time.size();

		csv.write(name);
		switch (outputDataType)
		{
		case SIZE:
			for (int i = 0; i < datasize; i++) csv.write(size[i]);
			break;
		case TIME_MEAN:
			for (int i = 0; i < datasize; i++) csv.write(time[i].x);
			break;
		case TIME_STD:
			for (int i = 0; i < datasize; i++) csv.write(time[i].y);
			break;
		case TIME_PIXEL_PER_MEDIAN:
			for (int i = 0; i < datasize; i++) csv.write(time[i].z / size[i] * 1000000.0);
			break;

		case TIME_MEDIAN:
		default:
			for (int i = 0; i < datasize; i++) csv.write(time[i].z);
			break;
		}
		csv.endline();
	}

	void show()
	{
		for (int i = 0; i < time.size(); i++)
		{
			cout << format("size: %8d mean: %10.4f std: %10.4f, med: %10.4f [ms] medperpix %.4f[ns]", size[i], time[i].x, time[i].y, time[i].z, time[i].z / size[i] * 1000000.0) << endl;
		}
		cout << endl;
	}
	void useOptimizedDetail()
	{
		if (cv::ipp::useIPP())cout << "enable IPP" << endl;
		else cout << "disable IPP" << endl;

		if (cv::useOptimized())cout << "enable optimized functions" << endl;
		else cout << "disable optimized functions" << endl;
	}

	Point3d counter(Mat& src, const int iteration)
	{
		cout << src.size() << endl;
		vector<double> v;

		Mat dest;
		cv::TickMeter meter;
		function(src, dest);//ignoring first function call

		for (int i = 0; i < iteration; i++)
		{
			meter.start();
			function(src, dest);
			meter.stop();
			v.push_back(meter.getTimeMilli());
		}

		double mean = 0.0;
		for (int i = 0; i < v.size(); i++) mean += v[i];
		mean /= v.size();

		double ver = 0.0;
		for (int i = 0; i < v.size(); i++) ver += (v[i] - mean)*(v[i] - mean);
		ver /= v.size();
		ver = sqrt(ver);

		double med = 0;
		std::sort(v.begin(), v.end());
		if (v.size() % 2 == 1) med = v[v.size() / 2];
		else med = (v[v.size() / 2 - 1] + v[v.size() / 2])*0.5;
		Point3d ret = Point3d(mean, ver, med);
		return  ret;
	}

	void run(Mat& src, int iteration)
	{
		Mat s;

		resize(src, s, Size(320, 240), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(480, 320), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(640, 480), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(1024, 768), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(1280, 960), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(1920, 1080), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(2880, 1620), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(3840, 2160), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(5760, 3240), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		resize(src, s, Size(7680, 4320), 0, 0);
		time.push_back(Point3d(counter(s, iteration)));
		size.push_back(s.size().area());

		show();
	}
};

class PerformanceTwoArg
{
public:
	vector<Point3d> time;
	vector<int> size;
	virtual void function(Mat& src1, Mat& src2, Mat& dest) = 0;

	enum
	{
		SIZE = -1,
		TIME_MEAN = 0,
		TIME_STD = 1,
		TIME_MEDIAN = 2,
		TIME_PIXEL_PER_MEDIAN = 3,
	};
	void csvout(CSV& csv, string name, int outputDataType = Performance::TIME_MEDIAN)
	{
		const int datasize = (int)time.size();

		csv.write(name);
		switch (outputDataType)
		{
		case SIZE:
			for (int i = 0; i < datasize; i++) csv.write(size[i]);
			break;
		case TIME_MEAN:
			for (int i = 0; i < datasize; i++) csv.write(time[i].x);
			break;
		case TIME_STD:
			for (int i = 0; i < datasize; i++) csv.write(time[i].y);
			break;
		case TIME_PIXEL_PER_MEDIAN:
			for (int i = 0; i < datasize; i++) csv.write(time[i].z / size[i] * 1000000.0);
			break;

		case TIME_MEDIAN:
		default:
			for (int i = 0; i < datasize; i++) csv.write(time[i].z);
			break;
		}
		csv.endline();
	}

	void show()
	{
		for (int i = 0; i < time.size(); i++)
		{
			cout << format("size: %8d mean: %10.4f std: %10.4f, med: %10.4f [ms] medperpix %.4f[ns]", size[i], time[i].x, time[i].y, time[i].z, time[i].z / size[i] * 1000000.0) << endl;
		}
		cout << endl;
	}
	void useOptimizedDetail()
	{
		if (cv::ipp::useIPP())cout << "enable IPP" << endl;
		else cout << "disable IPP" << endl;

		if (cv::useOptimized())cout << "enable optimized functions" << endl;
		else cout << "disable optimized functions" << endl;
	}

	Point3d counter(Mat& src1, Mat& src2, const int iteration)
	{
		vector<double> v;

		Mat dest;
		cv::TickMeter meter;
		function(src1, src2, dest);//ignoring first function call

		for (int i = 0; i < iteration; i++)
		{
			meter.start();
			function(src1, src2, dest);
			meter.stop();
			v.push_back(meter.getTimeMilli());
		}

		double mean = 0.0;
		for (int i = 0; i < v.size(); i++) mean += v[i];
		mean /= v.size();

		double ver = 0.0;
		for (int i = 0; i < v.size(); i++) ver += (v[i] - mean)*(v[i] - mean);
		ver /= v.size();
		ver = sqrt(ver);

		double med = 0;
		std::sort(v.begin(), v.end());
		if (v.size() % 2 == 1) med = v[v.size() / 2];
		else med = (v[v.size() / 2 - 1] + v[v.size() / 2])*0.5;
		Point3d ret = Point3d(mean, ver, med);
		return  ret;
	}

	void run(Mat& src1, Mat& src2, int iteration)
	{
		Mat resl;
		Mat resr;

		Size s = Size(320, 240);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(480, 320);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(640, 480);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(1024, 768);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(1280, 960);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(1920, 1080);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(2880, 1620);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(3840, 2160);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(5760, 3240);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		s = Size(7680, 4320);
		resize(src1, resl, s);
		resize(src2, resr, s);
		time.push_back(Point3d(counter(resl, resr, iteration)));
		size.push_back(src1.size().area());

		show();
	}
};

class Add : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		add(src, src, dest);
	}
};

class Mul : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		multiply(src, src, dest);
	}
};

class Div : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		divide(src, src, dest);
	}
};

class Threshold : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		threshold(src, dest, 30, 255, THRESH_BINARY);
	}
};

class CvtColorYUV : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		cvtColor(src, dest, COLOR_BGR2YUV);
	}
};

class CvtColorHSV : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		cvtColor(src, dest, COLOR_BGR2HSV);
	}
};

class Resize : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		Size size = Size(src.cols * 2, src.rows * 2);

		resize(src, dest, size, 0, 0, INTER_CUBIC);
	}
};

class BilateralFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		bilateralFilter(src, dest, 7, 30, 30, BORDER_REPLICATE);
	}
};

class Filter2D8u : public Performance
{
	Mat kernel;
	void function(Mat& src, Mat& dest)
	{
		filter2D(src, dest, CV_8U, kernel);
	}
public:
	Filter2D8u()
	{
		kernel.create(Size(7, 7), CV_8U);
		randu(kernel, 1, 20);
	}
};

class Filter2D32f : public Performance
{
	Mat kernel;
	void function(Mat& src, Mat& dest)
	{
		filter2D(src, dest, CV_32F, kernel);
	}
public:
	Filter2D32f()
	{
		kernel.create(Size(7, 7), CV_32F);
		randu(kernel, 0, 20);
	}
};

class GaussianFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		GaussianBlur(src, dest, Size(31, 31), 10.0);
	}
};

class DilateFilter : public Performance
{
	Mat kernel;
	void function(Mat& src, Mat& dest)
	{

		dilate(src, dest, kernel);
	}
public:
	DilateFilter()
	{
		kernel = Mat::ones(7, 7, CV_8U);
	}
};

class MedianFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		medianBlur(src, dest, 7);
	}
};

class BoxFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		boxFilter(src, dest, src.depth(), Size(7, 7));
	}
};

class FFT : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		cv::dft(src, dest);
	}
};

class EdgePreservingFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		edgePreservingFilter(src, dest);
	}
};

class Mean : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		Scalar v = mean(src);
	}
};

class MeanStd : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		Scalar mean;
		Scalar std;
		meanStdDev(src, mean, std);
	}
};

class Minmax : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
	}
};

class Histogram : public Performance
{
	Mat hist;
	void function(Mat& src, Mat& dest)
	{
		int binNum = 256;
		int histSize[] = { binNum };

		float value_ranges[] = { 0, 256 };
		const float* ranges[] = { value_ranges };
		MatND hist;
		int channels[] = { 0 };
		int dims = 1;

		calcHist(&src, 1, channels, Mat(),
			hist, dims, histSize, ranges,
			true,
			false);
	}
};

class CannyEdgeDetection : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		Canny(src, dest, 10, 20);
	}
};

class GuidedFilter : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		ximgproc::guidedFilter(src, src, dest, 13, 0.01);
	}
};

class SLIC : public Performance
{
	void function(Mat& src, Mat& dest)
	{
		Ptr<ximgproc::SuperpixelSLIC> a = ximgproc::createSuperpixelSLIC(src, ximgproc::SLIC);
		a->iterate(10);
		a->getLabels(dest);
	}
};

class StereoBlockMatching : public PerformanceTwoArg
{
	Ptr<StereoBM> sbm;
	void function(Mat& left, Mat& right, Mat& dest)
	{
		sbm->compute(left, right, dest);
	}
public:
	StereoBlockMatching()
	{
		sbm = StereoBM::create(256, 41);
	}
};

class FaceDetect : public Performance
{
	std::vector<Rect> faces;
	CascadeClassifier cascade;
	void function(Mat& src, Mat& dest)
	{
		cascade.detectMultiScale(src, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		/*
		//for debug
		// Draw circles on the detected faces
		for (int i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(src, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
		imshow("Detected Face", src);
		waitKey();
		*/
	}
public:
	FaceDetect()
	{
		cascade.load("haarcascade_frontalface_alt2.xml");
	}
};

int main(int argc, char** argv)
{
	CSV csv("OpenCVBenchmark_Median.csv", true, true);
	cout << getBuildInformation() << endl;
	Mat face = imread("face.jpg");
	Mat src = imread("8K.jpg");
	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat gray = imread("8K.jpg", 0);
	Mat grayf; gray.convertTo(grayf, CV_32F);

	Mat left = imread("im0.png", 0);
	Mat right = imread("im1.png", 0);

	cv::TickMeter meter;
	meter.start();

	cout << "======== Map ========" << endl;
	cout << "Add" << endl;
	Add ad;
	ad.run(gray, 100);
	ad.csvout(csv, "Size", Performance::SIZE);//for size index
	ad.csvout(csv, "add", Performance::TIME_MEDIAN);

	cout << "Add 32f" << endl;
	Add adf;
	adf.run(grayf, 100);
	adf.csvout(csv, "add 32f", Performance::TIME_MEDIAN);

	cout << "Mul" << endl;
	Mul mul;
	mul.run(gray, 100);
	mul.csvout(csv, "mul", Performance::TIME_MEDIAN);

	cout << "Mul 32f" << endl;
	Mul mulf;
	mulf.run(grayf, 100);
	mulf.csvout(csv, "mul 32f", Performance::TIME_MEDIAN);

	cout << "Div" << endl;
	Div di;
	di.run(gray, 100);
	di.csvout(csv, "div", Performance::TIME_MEDIAN);

	cout << "Div 32f" << endl;
	Div dif;
	dif.run(grayf, 100);
	dif.csvout(csv, "div 32f", Performance::TIME_MEDIAN);
	
	cout << "Threshold" << endl;
	Threshold th;
	th.run(gray, 100);
	th.csvout(csv, "threshold", Performance::TIME_MEDIAN);

	cout << "Threshold 32f" << endl;
	Threshold th32f;
	th32f.run(grayf, 100);
	th32f.csvout(csv, "threshold 32f", Performance::TIME_MEDIAN);

	cout << "CvtColor YUV" << endl;
	CvtColorYUV cvtcoloryuv;
	cvtcoloryuv.run(src, 100);
	cvtcoloryuv.csvout(csv, "cvtColor YUV", Performance::TIME_MEDIAN);

	cout << "CvtColor HSV" << endl;
	CvtColorHSV cvtcolorhsv;
	cvtcolorhsv.run(src, 100);
	cvtcolorhsv.csvout(csv, "cvtColor HSV", Performance::TIME_MEDIAN);

	cout << "======== Stencil ========" << endl;
	cout << "Filter2D 8u" << endl;
	Filter2D8u f2d8u;
	f2d8u.run(gray, 100);
	f2d8u.csvout(csv, "Filter2D 8u", Performance::TIME_MEDIAN);

	cout << "Filter2D 32f" << endl;
	Filter2D32f f2d32f;
	f2d32f.run(grayf, 100);
	f2d32f.csvout(csv, "Filter2D 32f", Performance::TIME_MEDIAN);

	cout << "Bilateral" << endl;
	BilateralFilter bf;
	bf.run(gray, 30);
	bf.csvout(csv, "Bilateral", Performance::TIME_MEDIAN);

	cout << "Bilateral 32f" << endl;
	BilateralFilter bf32f;
	bf32f.run(grayf, 30);
	bf32f.csvout(csv, "Bilateral 32f", Performance::TIME_MEDIAN);

	cout << "Bilateral Color" << endl;
	BilateralFilter bfc;
	bfc.run(src, 30);
	bfc.csvout(csv, "Bilateral Color", Performance::TIME_MEDIAN);

	cout << "Bilateral 32f" << endl;
	BilateralFilter bfc32f;
	bfc32f.run(srcf, 30);
	bfc32f.csvout(csv, "Bilateral Color 32f", Performance::TIME_MEDIAN);

	cout << "GaussianFilter" << endl;
	GaussianFilter gaussf;
	gaussf.run(gray, 30);
	gaussf.csvout(csv, "Gaussian", Performance::TIME_MEDIAN);

	cout << "GaussianFilter 32f" << endl;
	GaussianFilter gaussf32f;
	gaussf32f.run(gray, 30);
	gaussf32f.csvout(csv, "Gaussian 32f", Performance::TIME_MEDIAN);

	cout << "Resize (Cubic)" << endl;
	Resize res;
	res.run(gray, 100);
	res.csvout(csv, "Resize Cubic", Performance::TIME_MEDIAN);

	cout << "Median" << endl;
	MedianFilter mf;
	mf.run(gray, 100);
	mf.csvout(csv, "Median", Performance::TIME_MEDIAN);

	cout << "======== Scan ========" << endl;

	cout << "boxFilter" << endl;
	BoxFilter bfilter;
	bfilter.run(gray, 30);
	bfilter.csvout(csv, "Box", Performance::TIME_MEDIAN);

	cout << "boxFilter32f" << endl;
	BoxFilter bfilter32f;
	bfilter32f.run(grayf, 100);
	bfilter32f.csvout(csv, "Box 32f", Performance::TIME_MEDIAN);

	cout << "boxFilter Color" << endl;
	BoxFilter bfilterc;
	bfilterc.run(src, 30);
	bfilterc.csvout(csv, "Box Color", Performance::TIME_MEDIAN);

	cout << "boxFilter Color 32f" << endl;
	BoxFilter bfilterc32f;
	bfilterc32f.run(srcf, 30);
	bfilterc32f.csvout(csv, "Box Color 32f", Performance::TIME_MEDIAN);

	cout << "Dilate" << endl;
	DilateFilter df;
	df.run(gray, 30);
	df.csvout(csv, "Dilate", Performance::TIME_MEDIAN);
	
	cout << "Domain Transform Filter" << endl;
	EdgePreservingFilter dtf;
	dtf.run(src, 30);
	dtf.csvout(csv, "Domain Transform", Performance::TIME_MEDIAN);

	cout << "FFT" << endl;
	FFT ft;
	ft.run(grayf, 30);
	ft.csvout(csv, "FFT", Performance::TIME_MEDIAN);

	cout << "======== Reduction ========" << endl;
	cout << "Mean" << endl;
	Mean mean;
	mean.run(gray, 100);
	mean.csvout(csv, "Mean", Performance::TIME_MEDIAN);

	cout << "Mean Std" << endl;
	MeanStd meanstd;
	meanstd.run(gray, 100);
	meanstd.csvout(csv, "Mean Std", Performance::TIME_MEDIAN);

	cout << "MinMax" << endl;
	Minmax minmax;
	minmax.run(gray, 100);
	minmax.csvout(csv, "MinMax", Performance::TIME_MEDIAN);

	cout << "Histogram" << endl;
	Histogram hist;
	hist.run(gray, 100);
	hist.csvout(csv, "Histogram", Performance::TIME_MEDIAN);

	cout << "======== Total ========" << endl;
	cout << "Canny" << endl;
	CannyEdgeDetection ce;
	ce.run(gray, 30);
	ce.csvout(csv, "Canny", Performance::TIME_MEDIAN);

	cout << "Guided Filter" << endl;
	GuidedFilter gf;
	gf.run(src, 10);
	gf.csvout(csv, "Guided Filter", Performance::TIME_MEDIAN);

	cout << "SLIC" << endl;
	SLIC slic;
	slic.run(src, 10);
	slic.csvout(csv, "SLIC", Performance::TIME_MEDIAN);

	cout << "Stereo Block Matching" << endl;
	StereoBlockMatching sbm;
	sbm.run(left, right, 10);
	sbm.csvout(csv, "Stereo Block Matching", Performance::TIME_MEDIAN);

	cout << "Face Detection" << endl;
	FaceDetect fd;
	fd.run(face, 10);
	fd.csvout(csv, "Face Detection", Performance::TIME_MEDIAN);

	meter.stop();
	cout <<"Total time: "<< meter.getTimeSec() <<" [sec]"<< endl;
	return 0;
}