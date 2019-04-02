#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <complex>

using namespace std;
using namespace cv;

void addNoise(Mat &src,int type,double k1=0,double k2=1,int isshow=1)
/*
type: 噪声类型 0 高斯 1 椒盐
k1: 高斯 均值   椒盐 椒噪声密度
k2: 高斯 方差   椒盐 盐噪声密度
*/
{
	RNG rng;
	int row = src.rows;
	int col = src.cols;
	double gauss,sp,uniform;
	int temp;
	
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
		{
		switch(type)
		{
		case 0:
			temp=src.at<uchar>(i, j);
			gauss=rng.gaussian(k2)+k1;
			if(gauss>0)
				temp+=(int)(gauss+0.5);
			else
				temp+=(int)(gauss-0.5);				//类型转换用int不要用uchar（转成了字符）
			//避免超出范围
			if(temp<0)	temp=0;
			if(temp>255)	temp=255;
			break;
		case 1:
			uniform=rng.uniform((double)0,double(1));
			if(uniform<k1)//pepper
				temp=0;
			else if(uniform>1-k2)
				temp=255;
			else temp=src.at<uchar>(i, j);
			break;
		default:
			temp=src.at<uchar>(i, j);
		}	
		src.at<uchar>(i, j)=temp;
		}
	if(isshow)
	{
	imshow("image",src);
	waitKey(0);
	}
}

Mat spatialFilter(Mat src,int m,int n,int type,int Q=0)
{
	int row = src.rows;
	int col = src.cols;
	int mm=(m-1)/2,nn=(n-1)/2;
	Mat pad(row+mm*2,col+nn*2,CV_8UC1,Scalar(0));
	Mat out(row,col,CV_8UC1,Scalar(0));
	//补零
	for(int i=mm;i<row+mm;i++)
	{		
		uchar* data1=src.ptr<uchar>(i-mm);
		uchar* data2=pad.ptr<uchar>(i);
		for(int j=nn;j<col+nn;j++)
		{
			data2[j]=data1[j-nn];
		}
	}
	//滤波
	for(int i=mm;i<row+mm;i++)
	{		
		uchar* data0=out.ptr<uchar>(i-mm);
		for(int j=nn;j<col+nn;j++)
		{
		//初始化temp
		double temp,temp1;
		int pixels[m*n];
		switch(type)
		{
		case 0://算术均值
		case 2://谐波均值
		case 5://最大值
			temp=0;
			break;
		case 1://几何均值
			temp=1;
			break;
		case 3://逆谐波均值
			temp=0;temp1=0;
			break;
		case 6://最小值
			temp=300;
			break;
		case 7://中点
			temp=0;temp1=300;
			break;
		default:
			temp=0;temp1=0;
		}
		//遍历
		uchar* data1;
		for(int p=0;p<m;p++)
		{
		data1=pad.ptr<uchar>(i+p-mm);
		for(int q=0;q<n;q++)
		{
		switch(type)
		{
		case 0://算术均值
			temp+=data1[j+q-nn];
			break;
		case 1://几何均值
			temp=temp*data1[j+q-nn];
			break;
		case 2://谐波均值
			temp+=pow(data1[j+q-nn],-1);
			break;
		case 3://逆谐波均值
			temp+=pow(data1[j+q-nn],Q);
			temp1+=pow(data1[j+q-nn],Q+1);
			break;
		case 4://中值滤波
			pixels[p*n+q]=data1[j+q-nn];
			break;
		case 5://最大值
			if(data1[j+q-nn]>temp)
			{
			temp=data1[j+q-nn];
			}
			break;
		case 6://最小值
			if(data1[j+q-nn]<temp)
			{
			temp=data1[j+q-nn];
			}
			break;
		case 7://中点
			if(data1[j+q-nn]>temp)
			{
			temp=data1[j+q-nn];
			}
			if(data1[j+q-nn]<temp1)
			{
			temp1=data1[j+q-nn];
			}
			break;
		default:
			temp=0;temp1=0;
		}
		}
		}
		switch(type)
		{
		case 0://算术均值
			data0[j-nn]=temp/m/n;
			break;
		case 1://几何均值
			data0[j-nn]=pow((double)temp,(double)1/(double)(m*n));
			
			break;
		case 2://谐波均值
			data0[j-nn]=m*n/temp;
			break;
		case 3://逆谐波均值
			data0[j-nn]=temp1/temp;
			break;
		case 4://中值滤波
			for(int z=0;z<m*n;z++)
			{
				for(int y=0;y<m*n-z-1;y++)
				{
				if(pixels[y]>pixels[y+1])
				{
				temp=pixels[y+1];
				pixels[y+1]=pixels[y];
				pixels[y]=temp;
				}
				}
			}
			data0[j-nn]=pixels[(m*n+1)/2];
			break;
		case 5://最大值
			data0[j-nn]=temp;
			break;
		case 6://最小值
			data0[j-nn]=temp;
			break;
		case 7://中点
			data0[j-nn]=(temp+temp1)/2;
			break;
		default:
			temp=0;temp1=0;
		}
		
		}
	}
	return out;
}

Mat generateMotionFilter(Mat src,double a,double b,double T)
/*
	a: x方向位移总距离
	b: y方向位移总距离
	T: 曝光时间	
*/
{
	int row = src.rows;
	int col = src.cols;
	int P=2*row;
	int Q=2*col;
	Mat planes[2];
	Mat filter(P,Q,CV_32FC2,Scalar(0));
	split(filter,planes);
	complex<double> temp;
	int ii,jj;
	for(int i=0;i<P;i++)
	{
		float* data0 = planes[0].ptr<float>(i);				//指针索引更高效更安全
		float* data1 = planes[1].ptr<float>(i);
		for(int j=0;j<Q;j++)
		{
		ii=i-row;jj=j-col;
		if(a*ii+b*jj!=0)						//fu le ge da ck!!!
		{								//为什么不告诉我还有这种情况！！！AAAAAAA
		complex<double> e(0,M_PI*(ii*a+jj*b)*(-1));
		temp=T*sin(M_PI*(ii*a+jj*b))/(M_PI*(ii*a+jj*b));
		temp=temp*exp(e);
		data0[j]=(float)temp.real();
		data1[j]=(float)temp.imag();
		}
		else
		{
		data0[j]=(float)T;
		data1[j]=(float)0;
		}
		}
	}
	double min,max;
	Point minLoc,maxLoc;
	minMaxLoc(planes[0],&min,&max,&minLoc,&maxLoc);
	cout<<min<<"---"<<max<<"---"<<minLoc<<"---"<<maxLoc<<endl;
	merge(planes, 2, filter);
	
	magnitude(planes[0], planes[1], planes[0]);//计算幅度
	Mat mag = planes[0];
	mag += Scalar::all(1); 
	log(mag, mag);//取对数便于显示 
	normalize(mag, mag, 0, 1, CV_MINMAX);//标定
	imshow("image_generateMotionFilter",mag);
	waitKey(0);
	
	return filter;	
}

Mat generateWienerFilter(Mat src,double K,double a,double b,double T,int Ftype=0)
{
	int row = src.rows;
	int col = src.cols;
	int P=2*row;
	int Q=2*col;
	Mat planes[2];
	Mat filter(P,Q,CV_32FC2,Scalar(0));
	split(filter,planes);
	complex<double> temp,h;
	int ii,jj;
	//计算Laplace算子的傅里叶变换
	Mat p(P,Q,CV_32FC1,Scalar(0));//补零啊啊啊啊啊啊
	p.at<float>(0,0)=0;p.at<float>(0,1)=1;p.at<float>(0,2)=0;
	p.at<float>(1,0)=1;p.at<float>(1,1)=4;p.at<float>(1,2)=1;
	p.at<float>(2,0)=0;p.at<float>(2,1)=1;p.at<float>(2,2)=0;	
	
	Mat planes1[] = {Mat_<float>(p), Mat::zeros(p.size(), CV_32F)}; 
	merge(planes1, 2, p); //构造两个通道的输入矩阵，这里决定是进行一维DFT还是二维DFT
	Mat L;
	dft(p,L);
	Mat planes2[2];
	split(L,planes2);

	for(int i=0;i<P;i++)
	{
		float* data0 = planes[0].ptr<float>(i);
		float* data1 = planes[1].ptr<float>(i);
		ii=i-row;
		float* data2 = planes2[0].ptr<float>(i);
		float* data3 = planes2[1].ptr<float>(i);
		for(int j=0;j<Q;j++)
		{
		jj=j-col;//偏移量 移到中间	
		if(a*ii+b*jj!=0)
		{									
		complex<double> e(0,M_PI*(ii*a+jj*b)*(-1));
		h=T*sin(M_PI*(ii*a+jj*b))/(M_PI*(ii*a+jj*b));
		h=h*exp(e);
		}
		else
		{
		h=T;
		}//计算退化函数

		if(Ftype==0)
			temp=conj(h)/(abs(h)*abs(h)+K);
		else
		{
			complex<double> pp(data2[j],data3[j]);
			temp=conj(h)/(abs(h)*abs(h)+K*abs(pp)*abs(pp));
		}
		data0[j]=(float)temp.real();
		data1[j]=(float)temp.imag();
		}
	}
	double min,max;
	Point minLoc,maxLoc;
	minMaxLoc(planes[0],&min,&max,&minLoc,&maxLoc);
	cout<<min<<"---"<<max<<"---"<<minLoc<<"---"<<maxLoc<<endl;
	merge(planes, 2, filter);
	/*
	magnitude(planes[0], planes[1], planes[0]);//计算幅度
	Mat mag = planes[0];
	mag += Scalar::all(1); 
	log(mag, mag);//取对数便于显示 
	normalize(mag, mag, 0, 1, CV_MINMAX);//标定
	imshow("image_generateWienerFilter",mag);
	waitKey(0);
	*/
	return filter;	
}

Mat BLPF(Mat image,Mat filter,int isshow=1)

{
	int row = image.rows;
	int col = image.cols;
	//0填充 + 移到中心
	int P=2*row;
	int Q=2*col;
	Mat image_pad(P, Q, CV_32FC1,Scalar(0));
		//灰度图像的显示:使用CV_8UC1，0-255 使用CV_32FC1, 0-1
	int kk,temp;
	for(int i=0;i<P;i++)
		for(int j=0;j<Q;j++)
		{
		if((i+j)%2==0)
			kk=1;
		else
			kk=-1;
		if(i<row&&j<col)
			temp=image.at<uchar>(i, j);
		else
			temp=0;
		image_pad.at<float>(i, j)=temp*kk;			
		}	
	//DFT
	Mat planes[] = {Mat_<float>(image_pad), Mat::zeros(image_pad.size(), CV_32F)}; 
	merge(planes, 2, image_pad); //构造两个通道的输入矩阵，这里决定是进行一维DFT还是二维DFT
	Mat image_dft;
	dft(image_pad,image_dft);
	//显示傅里叶幅度谱
	split(image_dft, planes);  
	magnitude(planes[0], planes[1], planes[0]);//计算幅度
	Mat mag = planes[0];
	mag += Scalar::all(1); 
	log(mag, mag);//取对数便于显示 
	normalize(mag, mag, 0, 1, CV_MINMAX);//标定
	if(isshow)
	{	
	imshow("image_dft1",mag);
	waitKey(0);
	}

	double min,max;
	Point minLoc,maxLoc;
	minMaxLoc(mag,&min,&max,&minLoc,&maxLoc);
	cout<<min<<"---"<<max<<"---"<<minLoc<<"---"<<maxLoc<<endl;
	//计算傅里叶功率谱
	Mat power;
	multiply(mag,mag,power);
	
	//滤波
	split(image_dft, planes); 
	Mat fplanes[2],oplanes[2],iplanes[2];
	split(filter, fplanes); 
		//实部
	multiply(planes[0],fplanes[0],oplanes[0]);
	multiply(planes[1],fplanes[1],oplanes[1]);
	subtract(oplanes[0],oplanes[1],iplanes[0]);
		//虚部
	multiply(planes[0],fplanes[1],oplanes[0]);
	multiply(planes[1],fplanes[0],oplanes[1]);
	add(oplanes[0],oplanes[1],iplanes[1]);
	Mat image_after;
	merge(iplanes, 2, image_after);	
	//显示傅里叶幅度谱
	magnitude(iplanes[0], iplanes[1], iplanes[0]);//计算幅度
	mag = iplanes[0];
	mag += Scalar::all(1); 
	log(mag, mag);//取对数便于显示 
	normalize(mag, mag, 0, 1, CV_MINMAX);//标定
	
	
	//IDFT得到输出及其显示      
	Mat image_out;
	dft(image_after,image_out,DFT_INVERSE+DFT_REAL_OUTPUT);//只取实部，避免寄生复分量
	image_out=abs(image_out);
	minMaxLoc(image_out,&min,&max,&minLoc,&maxLoc);
	cout<<min<<"---"<<max<<"---"<<minLoc<<"---"<<maxLoc<<">>>>>>>>>>>>>"<<endl;
	normalize(image_out, image_out, 0, 1, CV_MINMAX);//标定			//float 必须在0-1之间才能正确显示
	Mat image_realout(image_out,Rect(0,0,row,col));//裁剪
	
	//float 0-1 映射到 uchar 0-255
	Mat image_got(row, col, CV_8UC1,Scalar(0));
	for(int i=0;i<row;i++)
	{
		uchar* data=image_got.ptr<uchar>(i);
		float* datar=image_realout.ptr<float>(i);
		for(int j=0;j<col;j++)
		{
		data[j]=(int)(datar[j]*255);					//必须用int作类型转换;后面必须加括号
		}
	}
	if(isshow)
	{	
	imshow("image_dft_filter2",mag);
	waitKey(0);
	imshow("image_realout3",image_realout);
	waitKey(0);
	}	
	return image_got;
}

void selectK(Mat image1)
{
	double K;
	while(1)
	{
	cout<<"Please input K:";
	cin>>K;
	if(K==-1)break;
	Mat filter=generateWienerFilter(image1,K,0.02,0.02,1,1);
	Mat image2=BLPF(image1,filter,0);
	imshow("last one",image2);
	waitKey(0);
	//imwrite("K.bmp",image2);
	}
}

void selectGamma(Mat image1,Mat Hfilter,double m,double sigma)
{
	double gamma=0.05;
	double a=10,lamda2;
	double study=0.01;
	int row = image1.rows;
	int col = image1.cols;
	lamda2=row*col*(sigma*sigma+m*m);
	while(1)
	{
	Mat filter=generateWienerFilter(image1,gamma,0.05,0.05,1,1);
	Mat image2=BLPF(image1,filter,0);
	//计算||r||2
	Mat image3=BLPF(image2,Hfilter,0);
	imshow("last one",image3);
	waitKey(0);
	double temp,rr=0;
	for(int i=0;i<row;i++)
	{
		uchar* data0=image1.ptr<uchar>(i);
		uchar* data1=image3.ptr<uchar>(i);
		uchar* data2=image2.ptr<uchar>(i);//是uchar不是float啊啊啊啊啊
		for(int j=0;j<col;j++)
		{
		temp=(int)data0[j]-(int)data1[j];
		rr+=temp*temp;
		}
	}
	if(abs(rr-lamda2)<=a)
	{
		imshow("last one",image2);
		waitKey(0);
		cout<<">>>>>>>>>>>>>"<<gamma<<"<<<<<<<<<<<<<<<<<"<<endl;
		break;
	}
	if(rr-lamda2<a*(-1))
		gamma+=study;
	if(rr-lamda2>a)
		gamma-=study;
	cout<<"*****"<<rr<<"****"<<lamda2<<"*****"<<gamma<<endl;
	}
	
}

int main()
{
	string file="/home/xyz/桌面/数字图像处理/作业/第6次作业/lena.bmp";
	Mat image=imread(file,0);
	if(!image.data)
	{
		printf("No image data\n");
		return -1;
	}
	//task1 2
	/*
	addNoise(image,0,0,20,0); 
	imshow("image",image);
	waitKey(0);
	imwrite("Gnoise.bmp",image);
	Mat image2=spatialFilter(image,3,3,3,1);//Mat spatialFilter(Mat src,int m,int n,int type,int Q=0)
	imshow("image2",image2);
	waitKey(0);
	imwrite("Gnoise31.bmp",image2);
	*/
	
	//task3	
	Mat Hfilter=generateMotionFilter(image,0.02,0.02,1);
	Mat image1=BLPF(image,Hfilter,0); 
	imwrite("lena_noise1.bmp",image1);
	addNoise(image1,0,0,10,0);
	imshow("image_noise",image1);
	waitKey(0);
	//imwrite("lena_noise.bmp",image1);
	//selectK(image1);
	//double m=0,sigma=10;
	
	//selectGamma(image1,Hfilter,m,sigma);

	return 0;
}
