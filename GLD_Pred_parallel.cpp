// GLD_Pred_parallel.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


double ((*G[7]))(double,double);
double G0(double x1, double x2){return 1.;}
double G1(double x1, double x2){return x1;}
double G2(double x1, double x2){return x2;}
double G3(double x1, double x2){return x1*x1;}
double G4(double x1, double x2){return x2*x2;}
double G5(double x1, double x2){return x1*x2;}

void GForming(){  // Forming of Functions G Array	
	G[0]=G0;  G[1]=G1; G[2]=G2; G[3]=G3; G[4]=G4; G[5]=G5; 
}

int n=5;// Number of the summands 
int m; // Implementation lengths
char c;
double *Y;        //Source time serial
double *w;        //WLDM weights
double	*p;       //GLDM weights
double	*Prgrad;  //Projection of the gradient
double	*z;       //WLDM approximation errors (The difference between actual and modelled values) 
double**P1;       //It is used for P calculation
double	**P;      //Projection matrix
double	**SST;    //Matrix for J-G transforming 
double *a1, *a;   //Identifyed parameters 
int  ri;          //The amount of basic equations of the primal problem
int *r;           //Ordinal numbers of the basic equations
double **PY;  // PY[i][t] is forward-looking forecast Y[i+t] 
int *FH;  // Reasonable Forecasting Horizons


void MemAlloc(){ // Memory allocation 
	w=new double [m+2]; for(int i=0; i<m+2; i++)w[i]=0.;
	p=new double [m+2]; for(int i=0; i<m+2; i++)p[i]=1.;
	Prgrad=new double [m+2]; for(int i=0; i<m+2; i++)Prgrad[i]=0.;
	z=new double [m+2]; for(int i=0; i<m+2; i++)z[i]=0.;
	P1=new double*[m+2]; for(int i=0; i<m+2; i++) P1[i]=new double[n+1];
	P=new double*[m+2];	for(int i=0; i<m+2; i++) P[i]=new double[m+2];
	SST=new double*[n+1]; for(int i=0; i<n+1; i++) SST[i]=new double [n+n+2];
	r=new int[n+1]; a1=new double [n+1];
	a=new double [n+1]; for(int i=0; i<=n; i++)a[i]=0.;
	PY=new double*[m+2]; for(int i=0; i<m+2; i++) PY[i]=new double[m+2];
	FH=new int[m+2];
}

//Forming of projecting matrix SST with adjoint matrix
void SSTForming(){
 #pragma omp parallel for shared(SST,G,Y)
for(int i=0; i<=n; i++){
	double A1, A2;
		for(int j=0; j<=n; j++){
			double SSTij=0;
			 //#pragma omp parallel reduction (+: SSTij)
			for(int t=3; t<=m; t++){
				A1=G[i](Y[t-2],Y[t-1]);
				A2=G[j](Y[t-2],Y[t-1]);
				SSTij+=A1*A2;
			}
			SST[i][j]=SSTij;
		}
		//Adding the adjoint matrix for Jordan-Gauss algorithm
		for(int j=1; j<=n; j++){
			SST[i][n+j]=0.;
		}
		SST[i][n+i]=1.;
	}
}
/*    //Draft printing 
		cout << '\n' << "Matrix SST" << '\n';
		for (int i=1; i<=n; i++){
			cout<< '\n' << i <<'\t';
			for(int j=1; j<=n+n; j++) cout << SST[i][j]  <<'\t';
		} 
*/

//Jordan-Gauss transforming
void JGTransforming(int nn){	
	for( int N=1; N<=nn; N++){
		//Find Lead Row
		int mm=N; // mm to find the lead row number
		double M=abs(SST[N][N]), Mi, M1=M;		
		#pragma omp parallel firstprivate(M1, mm) private(Mi)
		{
		#pragma omp for 
		for(int i=N+1; i<=n; i++){
			Mi=SST[i][N];
			if(abs(Mi)>M1)
				{mm=i; M1=Mi;}
		}
		#pragma omp critical
		{
			M=(M>M1)? M:M1;
		}
		}

		// Swapping of current N-th and lead mm-th rows
	#pragma omp parallel for shared (SST, mm, N)
	for(int K = 1; K<=2*n; K++) { //Swapping of lines cannot be parallelized :(
		double Temp = SST[N][K];
		SST[N][K]=SST[mm][K];
		SST[mm][K]=Temp;
	};

   // Normalise of the current row 
	double R = SST[N][N];
	#pragma omp parallel for shared(SST) (firstprivate R,N )
	for(int L=N; L<=nn; L++) {
		SST[N][L]/=R;
	}

  // Orthogonalize the Current Collumn 
	#pragma omp parallel for shared(SST) 
	for(int K=1; K<N; K++){
		if(K!=N) {
			double R = SST[K][N];
			for(int L=N; L<=nn; L++) 
				SST[K][L]-=SST[N][L]*R;
		} else continue;
	} //Iverting is ended
}
}

/*     
		//Draft printing 
		cout << '\n' << "Matrix SST^-1" << '\n';
		for (int i=1; i<=n; i++){
			cout<< '\n' << i <<'\t';
			for(int j=1; j<=n+n; j++) cout << SST[i][j]  <<'\t';
		} 
 */ 


void P1Forming(){ // Forming of Matrix P1 
	 #pragma omp parallel for shared (SST,G,Y) firstprivate(m,n)
	 for(int t=3; t<=m; t++){		
		for(int j=1; j<=n; j++) { 
			double p1=0.0;
			#pragma omp parallel for reduction(+:p1)
			for(int k=0; k<=n; k++){ 
				double A1=G[k](Y[t-2],Y[t-1]);				
				p1+=A1*SST[k][n+j];
			}
			P1[t][j]=p1;
		}
	}
} 
/* 
		cout << '\n' << "Matrix P1[3:m][1:n]" << '\n';
		for (int t=3; t<=m; t++){
			cout<< '\n' << t <<'\t';
			for(int j=1; j<=n; j++) cout << P1[t][j]  <<'\t';
		}
	
*/

void PForming(){// Forming the Projecting Matrix P[3:m][3:m] 
	#pragma omp parallel for shared (G,Y) firstprivate(m,n)
	for(int t1=3; t1<=m; t1++){
		for(int t2=3; t2<=m; t2++) { 
			double p1=0.0;
			#pragma omp parallel for reduction(+:p1)
			for(int j=0; j<=n; j++){ 
				double A1=G[j](Y[t2-2],Y[t2-1]);
				p1-=A1*P1[t1][j];
			} 	
			P[t1][t2]=p1;						   
   		}
		P[t1][t1]+=1.;
	}
/*
	cout << '\n' << "Matrix P[3:m][3:m]" << '\n';
	for (int i=3; i<=m; i++){
		cout<< '\n' << i <<'\t';
		for(int j=3; j<=m; j++) cout << P[i][j]  <<'\t';
	}
 */
}

void PrGradForming(){ //  find the gradient projection 
	#pragma omp parallel for shared(Prgrad,Y,P)
	for(int i=3; i<=m; i++){
		double S=0.0;
		#pragma omp parallel for reduction(+:S)
		for(int j=3; j<=m; j++)
			S+=P[i][j]*Y[j];
		Prgrad[i]=S;
	}	//  gradient projection is found 
/* 
	cout<< '\n' << "i   Y[i]   Prgrad[i]    p[i]  " << '\n';
	for (int i=3; i<=m; i++){
		cout<< '\n' << i <<'\t' << Y[i]  <<'\t'<< Prgrad[i]  <<'\t'<< p[i];
	}
*/
}


void DualWLDMSolution(){ //The function finds a solution to the dual problem w [t] and the number of active constraints
	double Al=LARGE;
	double Alc;
	int C=0;// the number of active constraints
	#pragma omp parallel for shared(w) firstprivate(m)
	for(int t=3; t<=m; t++)w[t]=0;
	#pragma omp parallel shared (Al,Prgrad,Alc,p,w)
	for(C=0; C<m-n-2; ){ //  Finding the Length of Moving along Prgrad
		Al=LARGE; //Al - offset length
		#pragma omp parallel shared (Al,Prgrad,p,w)
		for(int t=3; t<=m; t++){ //For each coordinate w [t], we narrow down the possible values of Al
		//p[t] - the given weights
		//w[t] - the variables of the dual problem		
			if(fabs(w[t])==p[t] ) 
			//w [t] takes a boundary value, we transfer it to the status of fixed ones
			continue;
			else{ 
				//In the occasional case of the input, the variable time is measured along the range of the gradient with the fixed t
				if(Prgrad[t]>0) Alc=(p[t]-w[t])/Prgrad[t]; 		
				else if(Prgrad[t]<0) Alc=(-p[t]-w[t])/Prgrad[t]; 				
				if(Alc<Al) Al=Alc; //offset length
			}
		}
		// After the cooling of the Al cycle, it contains the maximum offset
	  // Length of moving along Prgrad is equal to Al
	  #pragma omp parallel for shared (Al,Prgrad,p,w,C)
		for(int jj=3; jj<=m; jj++)
			if(fabs(w[jj])!=p[jj]){
				w[jj]+=Al*Prgrad[jj];//For points that are not needed on the boundary, 
				// we make a step of length Al in the direction of the graient
				if(fabs(w[jj])==p[jj])
					C++; //When the limit is lowered by 1 the number of the active constraints
			}
	} 
}

/* //Отладочная печать
	cout<< '\n' << " t    w[t]    p[t] " << '\n';
	for (int i=1; i<=m; i++){
		cout<< '\n' << i <<'\t'<< w[i] << '\t'<< p[i] ;
	} 
*/

void PrimalWLDMSolution(){
	ri=0; //ri equal to number of the basic equations
	#pragma omp parallel for shared (ri)
	for(int t=3; t<=m; t++){	
		if( fabs(w[t])!=p[t]) {
			++ri; 
			r[ri]=t;
		}
	}
	#pragma omp parallel for shared(Y,G,r,SST) firstprivate(ri)
	for(int l=1; l<=ri; l++) {  //  Formig of the Equations 
			//double Yr=Y[r[i]];
			for(int i=0; i<=ri; i++){			
				double A1=G[i](Y[r[l]-1],Y[r[l]-2]);		
				SST[l][i]=A1;
			}
			SST[l][ri+1]=Y[r[l]];
	}		
	JGTransforming( ri);
	#pragma omp parallel for shared(SST,a,z) firstprivate(ri)
	for(int i=1; i<=ri; i++) {
		a[i]=SST[i][ri+1];
		z[r[i]]=0;
	}
/*	cout<< '\n';
	for(int i=1; i<=n; i++)
		cout <<"a["<<i<<"]="<<a[i]<<'\t';
	cout<< '\n'; 
	*/
}

double GLDMEstimator(){
	SSTForming();
	JGTransforming(n);
	P1Forming(); PForming(); PrGradForming();

	double Z, d;
	do{
		#pragma omp parallel for
		   for(int ii=1; ii<=n; ii++) a1[ii]=a[ii];
		#pragma omp parallel for 
		   for(int ii=1; ii<=m; ii++) p[ii]=1./(1.+z[ii]*z[ii]);
		#pragma omp parallel for
		   for(int ii=1; ii<=m; ii++) w[ii]=0.;
		DualWLDMSolution(); //Solution of dual problem
		PrimalWLDMSolution(); //Solution of primal problem
		Z=z[1]=z[2]=0.; //Defining the residuals
		//Z be the loss function
		// z be the vctor of residuals
		double A1;
		#pragma omp parallel for reduction(+:Z) //
		for(int t=3; t<=m; t++) {
			//z[t]=Y[t];
			double zr=Y[t];				
			#pragma omp parallel for reduction(+:zr) 
			for(int i=0; i<=n; i++){
				A1=G[i](Y[t-1],Y[t-2]);
				zr-=a[i]*A1;
			} 
			z[t]=zr;
			Z+=fabs(z[t]);
		}
		d=fabs(a[1]-a1[1]);
		#pragma omp parallel for shared(d)
		for(int i=2; i<=n; i++) 
			if(d<fabs(a[i]-a1[i])) d=fabs(a[i]-a1[i]);

	}while(d);
	return Z;
}

double	E;
double	D;
int minFH; //reasonable forecasting horizon
double LastStrt;
double SZ;

void ForecastingEst(){ // Calculation of the average prediction errors

	
	int t,   // forecasting horizon (time horizon)
		 Strt=0,	 // St - start point, Et=St+T-1 - end point of the forecasting interval
		 Et;  // Et=St+T-1 - end point of the forecasting interval
#pragma omp parallel for 	
	do{
		Strt++; 			
		PY[Strt][0]=Y[Strt]; 
		PY[Strt][1]=Y[Strt+1]; 
		for(t=Strt+2; t<m; t++){
			double py=0; 
			#pragma omp parallel for reduction(+: py)
			for(int j=0; j<=n; j++){
				double A1=G[j](PY[Strt][t-1],PY[Strt][t-2]);
				double R=a[j]*A1;
				py+=R;
			}
			PY[Strt][t]=py; 
			if(fabs(PY[Strt][t]-Y[(Strt)+t]) > SZ) break;
			}
		FH[Strt]=t; // FH[Strt] - reliablel forecasting horizon from Strt
	}while(FH[Strt]<m);
	LastStrt=t; 

//		cout<< '\n' << " Strt="<< Strt << "  FH=[Strt]= " << t << '\n';

	// Now Strt equal to number of the used fragments

	//find minimal FH[t] for all t<Strt
	minFH=FH[Strt];
	#pragma omp parallel for private (minFHp)
	{
		int minFHp=minFH;
		for (int t=3;t<Strt;t++){
		if (FH[t]<minFH) minFHp=FH[t];
		}
		#pragma omp critical 
		minFH=(minFHp<minFH)? minFHp : minFH;
	}
	// Now minFH is equal to reasonable forecasting horizon 

		E=D=0;
		#pragma omp parallel for shared(Y,PY) reduction(+:D,E) 
		for(int t=3; t<=minFH; t++){
				D+=fabs(Y[t+Strt]-PY[Strt][t]); // The summ of absolute errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
				E+=(Y[t+Strt]-PY[Strt][t]);	// The summ of errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
			}
		D/=minFH; E/=minFH; // The average errors of the prediction for time horizon minFH
			
}

int main()
{
fstream f("Data.txt", ios::in); //Data.txt
		// Data Input 
		char c; 
		do f>>c; 
			while (c!=':'); 
		f>>m;
		Y=new double [m+2];
		double s, sm=0.;
		for(int ic=1; ic<=m; ic++){
			f>>s; Y[ic]=s;	sm=(sm<s)?s:sm;
		} 
		f.close();
		for(int ic=1; ic<=m; ic++)Y[ic]/=sm;
		//End of Data Input 

	MemAlloc();	 // Memory allocation 
	// Output
	char FileName[8]="Out.txt";
   // sprintf(FileName, "%d", t);
	ofstream g(FileName);
   	g.setf(ios_base::scientific);
//-----------------------------------------------------------------------------
double start, stop; //start and stop time of working of the algorithm	
int maxThread=8; //omp_get_max_threads(); //maximal number of threads available. For my laptop it is =4, AV = 8
//int maxThread=1;
for (int t=1;t<=maxThread;t++){ //we held our experiment for 1,2,...,20 threads
  omp_set_num_threads(t);
    start = omp_get_wtime();
	GForming();  // Forming of Functions G Array
	// Solution
	double SZ=GLDMEstimator(); //Procedure of estimating using Generalized Least Deviation Method  
	//----------------------------
	stop = omp_get_wtime();	
	//-----------------------------
	//Output
/*   	g<<"Optimal factors : ";
   	for(int i=0; i<=n; i++)g<<'\n'<<"a["<<i<<"]="<<a[i];
	g<<'\n'<<'\n'<<"Optimal value of Loss function : "<<SZ<<'\n';
*/
ForecastingEst(); // Calculation of the average prediction errors
g.precision(9); 
g<<'\n'<<"Reasonable forecasting horizon = "<<minFH;
// g<<'\n'<<"Average prediction errors: E="<< E <<"; D=" << D;
g<<"\nThreads: "<<t<<"\t Time:"<<stop-start; //The number of threads + time need for calculations
g<<"\n*****************************************************\n";
	//  Completion
	
}//for the number of threads----------------------------------------------------------
g.close();
L1: cout<<'\n'<<"Press any key:  ";	
	getchar(); 
	return(0);
}
