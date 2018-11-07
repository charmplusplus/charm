// This program solves the 2-d wave equation over a grid.

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "mpi.h"

#if USE_LIVEVIZ
#include "liveViz.h"
#endif

#define NumIters        5000
#define TotalDataWidth  800
#define TotalDataHeight 800
#define NumInitPerturbs 5

#define nbor(a,b) (a*comm_dim+b)
#define mod(a,b)  (((a)+b)%b)

struct indexStruct { int x, y; };
enum { left=0, right, up, down };

int main(int argc, char **argv) {
  int my_rank, comm_size, num_wths, flag;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const int comm_dim = round(sqrt(comm_size));
  if (comm_size != comm_dim * comm_dim) {
    if (my_rank == 0) {
      printf("Error: wave2d must be run with a perfect square number of VPs!\n");
    }
    MPI_Finalize();
    return 1;
  }
  else if (my_rank == 0) {
    MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_NUM_WTHS, &num_wths, &flag);
    printf("Running wave2d on %d VPs on %d processors\n", comm_size, num_wths);
  }

#if USE_LIVEVIZ
  // Setup liveviz
  CkArrayOptions opts(comm_size);
  CkCallback cb(CkIndex_Wave::requestNextFrame(NULL), arrayProxy);
  liveVizConfig cfg(liveVizConfig::pix_color, true);
  liveVizInit(cfg, arrayProxy, cb, opts);
#endif

  const int my_x = my_rank / comm_dim;
  const int my_y = my_rank % comm_dim;
  const int num_nbors = 4;
  const int my_width  = TotalDataWidth / comm_dim;
  const int my_height = TotalDataHeight / comm_dim;
  const int left_nbor  = nbor(mod(my_x-1, comm_dim), my_y);
  const int right_nbor = nbor(mod(my_x+1, comm_dim), my_y);
  const int down_nbor  = nbor(my_x, mod(my_y-1, comm_dim));
  const int up_nbor    = nbor(my_x, mod(my_y+1, comm_dim));
  double *pressure_old = new double[my_width*my_height]; // time t-1
  double *pressure     = new double[my_width*my_height]; // time t
  double *pressure_new = new double[my_width*my_height]; // time t+1
  double *buffers[num_nbors];
  buffers[left]  = new double[my_height];
  buffers[right] = new double[my_height];
  buffers[up]    = new double[my_width];
  buffers[down]  = new double[my_width];
  double *left_edge  = new double[my_height];
  double *right_edge = new double[my_height];
  MPI_Request request[num_nbors*2];

  // Setup some Initial pressure pertubations for timesteps t-1 and t
  srand(0); // Force the same random numbers to be used for each rank
  for (int i = 0; i < my_height*my_width; i++) { pressure[i] = pressure_old[i] = 0.0; }
  for (int s = 0; s < NumInitPerturbs; s++) {
    // Determine where to place a circle within the interior of the 2-d domain
    int radius  = 20 + rand() % 30;
    int xcenter = radius + rand() % (TotalDataWidth - 2*radius);
    int ycenter = radius + rand() % (TotalDataHeight - 2*radius);
    // Draw the circle
    for (int i = 0; i < my_height; i++) {
      for (int j = 0; j < my_width; j++) {
        int globalx = my_x*my_width + j; // The coordinate in the global data array (not just in this rank's portion)
        int globaly = my_y*my_height + i;
        double distanceToCenter = sqrt((globalx-xcenter)*(globalx-xcenter) + (globaly-ycenter)*(globaly-ycenter));
        if (distanceToCenter < radius) {
          double rscaled = (distanceToCenter/radius)*3.0*3.14159/2.0; // ranges from 0 to 3pi/2
          double t = 700.0 * cos(rscaled) ; // Range won't exceed -700 to 700
          pressure[i*my_width+j] = pressure_old[i*my_width+j] = t;
        }
      }
    }
  }

  for (int iter = 0; iter < NumIters; iter++) {
    // Exchange edge buffers with neighboring ranks
    MPI_Irecv(buffers[right], my_height, MPI_DOUBLE, right_nbor, right, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(buffers[left],  my_height, MPI_DOUBLE, left_nbor,  left,  MPI_COMM_WORLD, &request[1]);
    MPI_Irecv(buffers[up],    my_width,  MPI_DOUBLE, up_nbor,    up,    MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(buffers[down],  my_width,  MPI_DOUBLE, down_nbor,  down,  MPI_COMM_WORLD, &request[3]);

    for (int i = 0; i < my_height; i++) { left_edge[i]  = pressure[i*my_width]; }
    MPI_Isend(left_edge, my_height, MPI_DOUBLE, left_nbor, right, MPI_COMM_WORLD, &request[4]);
    for (int i = 0; i < my_height; i++) { right_edge[i] = pressure[i*my_width + my_width-1]; }
    MPI_Isend(right_edge, my_height, MPI_DOUBLE, right_nbor, left, MPI_COMM_WORLD, &request[5]);
    double *top_edge = &pressure[0];
    MPI_Isend(top_edge, my_width, MPI_DOUBLE, up_nbor, down, MPI_COMM_WORLD, &request[6]);
    double *bottom_edge = &pressure[(my_height-1)*my_width];
    MPI_Isend(bottom_edge, my_width, MPI_DOUBLE, down_nbor, up, MPI_COMM_WORLD, &request[7]);

    MPI_Waitall(num_nbors*2, request, MPI_STATUSES_IGNORE);

    for (int i = 0; i < my_height; i++) {
      for (int j = 0; j < my_width; j++) {
        // Current time's pressures for neighboring array locations
        double L = (j==0          ? buffers[left][i]  : pressure[i*my_width+j-1]);
        double R = (j==my_width-1  ? buffers[right][i] : pressure[i*my_width+j+1]);
        double U = (i==0          ? buffers[up][j]    : pressure[(i-1)*my_width+j]);
        double D = (i==my_height-1 ? buffers[down][j]  : pressure[(i+1)*my_width+j]);
        // Current time's pressure for this array location
        double curr = pressure[i*my_width+j];
        // Previous time's pressure for this array location
        double old  = pressure_old[i*my_width+j];
        // Compute the future time's pressure for this array location
        pressure_new[i*my_width+j] = 0.4*0.4*(L+R+U+D - 4.0*curr)-old+2.0*curr;
      }
    }

    // Advance to next step by shifting the data back one step in time
    double *tmp = pressure_old;
    pressure_old = pressure;
    pressure = pressure_new;
    pressure_new = tmp;

    //MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0 && iter % 20 == 0) {
      printf("Completed %d iterations\n", iter);
    }
#if CMK_MEM_CHECKPOINT
    if (iter != 0 && iter % 200 == 0) {
      AMPI_Migrate(AMPI_INFO_CHKPT_IN_MEMORY);
    }
#endif
  }

  // Clean up and exit
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) printf("Program Done!\n");
  delete [] pressure_new;
  delete [] pressure;
  delete [] pressure_old;
  delete [] buffers[left];
  delete [] buffers[right];
  delete [] buffers[up];
  delete [] buffers[down];
  delete [] right_edge;
  delete [] left_edge;
  MPI_Finalize();
  return 0;
}

#if USE_LIVEVIZ
// Provide my portion of the image to the graphical liveViz client
void requestNextFrame(liveVizRequestMsg *m) {
  // Draw my part of the image, plus a nice 1px border along my right/bottom boundary
  int sx = my_x*my_width; // Where my portion of the image is located
  int sy = my_y*my_height;
  int w = my_width; // Size of my rectangular portion of the image
  int h = my_height;

  // Set the output pixel values for my rectangle
  // Each RGB component is a char which can have 256 possible values.
  unsigned char *intensity = new unsigned char[3*w*h];
  for (int i = 0; i < my_height; i++) {
    for (int j = 0; j < my_width; j++) {
      double p = pressure[i*my_width+j];
      if (p > 255.0) p = 255.0;   // Keep values in valid range
      if (p < -255.0) p = -255.0; // Keep values in valid range

      if (p > 0) { // Positive values are red
        intensity[3*(i*w+j)+0] = 255;   // RED component
        intensity[3*(i*w+j)+1] = 255-p; // GREEN component
        intensity[3*(i*w+j)+2] = 255-p; // BLUE component
      } else { // Negative values are blue
        intensity[3*(i*w+j)+0] = 255+p; // RED component
        intensity[3*(i*w+j)+1] = 255+p; // GREEN component
        intensity[3*(i*w+j)+2] = 255;   // BLUE component
      }
    }
  }

  // Draw a green border on right and bottom of this rank's pixel buffer.
  // This will overwrite some pressure values at these pixels.
  for (int i = 0; i < h; i++) {
    intensity[3*(i*w+w-1)+0] = 0;     // RED component
    intensity[3*(i*w+w-1)+1] = 255;   // GREEN component
    intensity[3*(i*w+w-1)+2] = 0;     // BLUE component
  }
  for (int i = 0; i < w; i++) {
    intensity[3*((h-1)*w+i)+0] = 0;   // RED component
    intensity[3*((h-1)*w+i)+1] = 255; // GREEN component
    intensity[3*((h-1)*w+i)+2] = 0;   // BLUE component
  }

  liveVizDeposit(m, sx, sy, w, h, intensity, this);
  delete [] intensity;
}
#endif
