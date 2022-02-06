// GLDM_Pred.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
using namespace std;

double ((*G[6]))(double,double);

double G1(double x1, double x2){ 
	return x1; 
}
double G2(double x1, double x2){ 
	return x2;
}
double G3(double x1, double x2){ 
	return x1*x1;
}
double G4(double x1, double x2){ 
	return x2*x2;
}
double G5(double x1, double x2){ 
	return x1*x2;
}

void GForming(){  // Forming of Functions G Array	
	G[1]=G1; G[2]=G2; G[3]=G3; G[4]=G4; G[5]=G5; 
}

int n=5;// Number of the summands 
int m; // Implementation lengths
char c;
fstream f("Data.txt", ios::in);

double *Y, *w, *p, *grad, *Prgrad, *z, **P1, **P, **SST; 
double *a1, *a; int *r, ri;
double **PY, *E, *D;

void MemAlloc(){ // Memory allocation 
	w=new double [m+2]; for(int i=0; i<m+2; i++)w[i]=0.;
	p=new double [m+2]; for(int i=0; i<m+2; i++)p[i]=1.;
	grad=new double [m+2]; for(int i=1; i<m+3; i++) grad[i]=Y[i];
	Prgrad=new double [m+2]; for(int i=0; i<m+2; i++)Prgrad[i]=0.;
	z=new double [m+2]; for(int i=0; i<m+2; i++)z[i]=0.;
	P1=new double*[m+2]; for(int i=0; i<m+2; i++) P1[i]=new double[n+1];
	P=new double*[m+2];	for(int i=0; i<m+2; i++) P[i]=new double[m+2];
	SST=new double*[n+1]; for(int i=0; i<=n; i++) SST[i]=new double [n+n+2];
	r=new int[n+1]; a1=new double [n+1];
	a=new double [n+1]; for(int i=0; i<=n; i++)a[i]=0.;
}

void SSTForming(){
for(int i=1; i<=n; i++){
		for(int j=1; j<=n; j++){
			SST[i][j]=0.;
			for(int t=3; t<=m; t++){
				double A1=G[i](Y[t-2],Y[t-1]);
				double A2=G[j](Y[t-2],Y[t-1]);
				SST[i][j]+=A1*A2;
			}
		}
		for(int j=1; j<=n; j++){
			SST[i][n+j]=0.;
		}
		SST[i][n+i]=1.;
	}
}

void JGTransforming(int nn){
	for( int N=1; N<=nn; N++){
		//Find Lead Row
		int mm=N; // mm to find the lead row number
		double Mi;
		double M=abs(SST[N][N]);
		for(int i=N+1; i<=nn; i++) 
			if(abs(Mi=SST[i][N])>M)
				{mm=i; M=Mi;}	 
	
		// Swapping of current N-th and lead mm-th rows
		double Temp;
	for(int K = 1; K<=2*nn; K++) {
		Temp = SST[N][K];
		SST[N][K]=SST[mm][K];
		SST[mm][K]=Temp;
	};

   // Normalise of the current row 
	double R = SST[N][N];
	for(int L=N; L<=2*nn; L++) {
		SST[N][L]/=R;
	}

  // Orthogonalize the Current Collumn
	for(int K=1; K<N; K++) {
		double R = SST[K][N];
		for(int L=N; L<=2*n; L++) 
			SST[K][L]-=SST[N][L]*R;
		}
		for(int K=N+1; K<=n; K++) {
		double R = SST[K][N];
		for(int L=N; L<=2*nn; L++) 
		  SST[K][L]-=SST[N][L]*R;
		}

	/* */
		cout << '\n' << "Matrix SST^-1" << '\n';
		for (int i=1; i<=nn; i++){
			cout<< '\n' << i <<'\t';
			for(int j=1; j<=nn+nn; j++) cout << SST[i][j]  <<'\t';
		} 
	} //Iverting is ended
}

void P1Forming(){ // Forming of Matrix P1
 	for(int t=3; t<=m; t++){		
		for(int j=1; j<=n; j++) { 
			P1[t][j]=0;
			for(int k=1; k<=n; k++){ 
				double A1=G[k](Y[t-2],Y[t-1]);				
				P1[t][j]+=A1*SST[k][n+j];
			}
		}
	}
/* */
		cout<< '\n' << "Matrix P1[3:m][1:n]" << '\n';
		for (int t=3; t<=m; t++){
			cout<< '\n' << t <<'\t';
			for(int j=1; j<=n; j++) cout << P1[t][j]  <<'\t';
		}
	}

void PForming(){// Forming the Projecting Matrix P[3:m][3:m] 
	for(int t1=3; t1<=m; t1++){
		for(int t2=3; t2<=m; t2++) { 
			P[t1][t2]=0;
			for(int j=1; j<=n; j++){ 
				double A1=G[j](Y[t2-2],Y[t2-1]);
				P[t1][t2]-=A1*P1[t1][j];
			} 							   
   		}
		P[t1][t1]+=1.;
	}
/* */
	cout<< '\n' << "Matrix P[3:m][3:m]" << '\n';
	for (int i=3; i<=m; i++){
		cout<< '\n' << i <<'\t';
		for(int j=3; j<=m; j++) cout << P[i][j]  <<'\t';
	}
}

void PrGradForming(){ //  find the gradient projection 
	for(int i=3; i<=m; i++){
		Prgrad[i]=0.;
		for(int j=3; j<=m; j++)
			Prgrad[i]+=P[i][j]*grad[j];
	}	//  gradient projection is found 

/* */
	cout<< '\n' << "i   grad[i]   Prgrad[i]    p[i]  " << '\n';
	for (int i=3; i<=m; i++){
		cout<< '\n' << i <<'\t' << grad[i]  <<'\t'<< Prgrad[i]  <<'\t'<< p[i];
	}
}

void DualWLDMSolution(){ 
	double Al=LARGE;
	double Alc;
	int C=0;  //  Nunber of aktive restrictions
	for(int t=3; t<=m; t++)w[t]=0;
	for(C=0; C<m-n-2; ){
		Al=LARGE;	 //  Finding the Length of Moving along Prgrad
		for(int t=3; t<=m; t++){		
			if(fabs(w[t])==p[t]) 
			continue;
			else{ 
				if(Prgrad[t]>0) Alc=(p[t]-w[t])/Prgrad[t]; 		
				else if(Prgrad[t]<0) Alc=(-p[t]-w[t])/Prgrad[t]; 				
				if(Alc<Al) Al=Alc;
			}
		}
	  // Length of Moving along Prgrad equal Al
		for(int jj=3; jj<=m; jj++)
			if(fabs(w[jj])!=p[jj]){
				w[jj]+=Al*Prgrad[jj];
				if(fabs(w[jj])==p[jj])
					C++; 
			}
	} 
	/* */
	cout<< '\n' << " t    w[t]    p[t] " << '\n';
	for (int i=1; i<=m; i++){
		cout<< '\n' << i <<'\t'<< w[i] << '\t'<< p[i] ;
	} 
}


void PrimalWLDMSolution(){
	ri=0; //ri equal to number of the basic equations
	for(int t=3; t<=m; t++){	
		if( fabs(w[t])!=p[t]) {
			++ri; 
			r[ri]=t;
		}
	}
	for(int l=1; l<=ri; l++) {  //  Formig of the Equations 
			//double Yr=Y[r[i]];
			for(int i=1; i<=ri; i++){			
				double A1=G[i](Y[r[l]-1],Y[r[l]-2]);		
				SST[l][i]=A1;
			}
			SST[l][ri+1]=Y[r[l]];
	}		
	JGTransforming( ri);
	for(int i=1; i<=ri; i++) {
		a[i]=SST[i][ri+1];
		z[r[i]]=0;
	}
	cout<< '\n';
	for(int i=1; i<=n; i++)
		cout <<"a["<<i<<"]="<<a[i]<<'\t';
	cout<< '\n'; 
}

double GLDMEstimator(){
	SSTForming();
	JGTransforming(n);
	P1Forming(); PForming(); PrGradForming();

	double Z, d;
	do{
		for(int ii=1; ii<=n; ii++) a1[ii]=a[ii];
		for(int ii=1; ii<=m; ii++) p[ii]=1./(1.+z[ii]*z[ii]);
		for(int ii=1; ii<=m; ii++) w[ii]=0.;
		DualWLDMSolution();
		PrimalWLDMSolution();
		Z=z[1]=z[2]=0.;
		for(int t=3; t<=m; t++) {
			z[t]=Y[t];				
			for(int i=1; i<=n; i++){
				double A1=G[i](Y[t-1],Y[t-2]);
				z[t]-=a[i]*A1;
			} 
			Z+=fabs(z[t]);
		}
		d=fabs(a[1]-a1[1]);
		for(int i=2; i<=n; i++) 
			if(d<fabs(a[i]-a1[i])) d=fabs(a[i]-a1[i]);

	}while(d);
	return Z;
}

	
void ForeñastingEst(){ // Calculation of the average prediction errors
	PY=new double*[m+2];
	E=new double[m+2];
	D=new double[m+2];
	for(int i=0; i<m+2; i++) PY[i]=new double[m+2];
	int T;   // forecasting horizon (time horizon)
	int St;	 // St - start point, Et=St+T-1 - end point of the forecasting interval
	int Et;  // Et=St+T-1 - end point of the forecasting interval
	for(T=1; T<=m-3; T++ ){
		E[T]=D[T]=0;
		for(St=3, Et=St+T-1; St<m-T; St++,Et++){
			PY[T][St-2]=Y[St-2]; PY[T][St-1]=Y[St-1];
			for(int i=St; i<=Et; i++){
				PY[T][i]=0;
				for(int j=1; j<=n; j++){
					double A1=G[j](PY[T][i-1],PY[T][i-2]);
					PY[T][i]+=a[j]*A1;
				}
				D[T]+=fabs(Y[i]-PY[T][i]); // The summ of absolute errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
				E[T]+=(Y[i]-PY[T][i]);	// The summ of errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
			}
		D[T]/=T; E[T]/=T; // The average errors of the prediction for time horizon T
	}
}
	
	
}

int main()
{
	{	// Data Input 
		char c; 
		do f>>c; 
			while (c!=':'); 
		f>>m;
		Y=new double [m+2];
/*		do f>>c; 
			while (c!=':'); */

		double s;
		for(int ic=1; ic<=m; ic++){
			f>>s; Y[ic]=s;		
		} 
		f.close();
	}	//End of Data Input 
	MemAlloc();	 // Memory allocation 
	GForming();  // Forming of Functions G Array
	// Solution
	SSTForming();
	JGTransforming(n);
	double SZ=GLDMEstimator();
	// Output
	ofstream g("Out.txt");
   	g.setf(ios_base::scientific);
   	g<<"Optimal factors : ";
   	for(int i=1; i<=n; i++)g<<'\n'<<"a["<<i<<"]="<<a[i];
	g<<'\n'<<'\n'<<"Optimal value of Loss function : "<<SZ<<'\n';
	ForeñastingEst(); // Calculation of the average prediction errors
	g<<'\n'<<"Average prediction errors";
	g<<'\n'<<" T '\t'    E '\t'      D "<< '\n';
	for( int T=3; T<m; T++)  g<<T<<'\t'<< E[T]<<'\t'<< D[T] <<'\n';
	//  Completion
	g.close();
L1: cout<<'\n'<<"Press any key:  ";	
	getchar(); 
	return(0);
}

