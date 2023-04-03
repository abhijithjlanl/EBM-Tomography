#include "itensor/all.h"
#include <stdlib.h>
//...
using namespace itensor;

int main(int argc, char **argv)
{
  float tau;
  int N;
  N = atoi(argv[1]);
  tau = atof(argv[2]);
  println("N = ", N);
  println("tau = ", tau);
  auto sites = SpinHalf(N,{"ConserveQNs=",false});
  auto ampo = AutoMPO(sites);
  //Make the Heisenberg Hamiltonian
  //        for(int b = 1; b < N; ++b)
  //        {
  //        ampo += 0.5,"S+",b,"S-",b+1;
  //        ampo += 0.5,"S-",b,"S+",b+1;
  //        ampo +=     "Sz",b,"Sz",b+1;
  //        }

  for(int b = 1; b < N; ++b)
    {
      ampo += 2,"Sx",b;
      ampo += -4,"Sz",b,"Sz",b+1;
    }
  ampo +=  2,"Sx", N;
  auto H = toMPO(ampo);
  auto expH = toExpH(ampo,tau);
  PrintData(expH);
  writeToFile("sites_file",sites);
  writeToFile("psi_file",expH);
  auto fo = h5_open("test.h5",'w');
  h5_write(fo,"expH",expH);
  close(fo);
  println("Done!" );
  return 0;
}
