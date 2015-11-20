/*
  particle_system.cpp
  Austin Brennnan
  10/8/2015

  This particle sytem with collisions uses velocity Verlet integration.
*/ 

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>


#define YAXIS 1
#define BOX 2
#define WALL 3
#define BORDER 4
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080


GLuint OGL_VBO = 1; // vertex buffer object
#define NUMBER_OF_PARTICLES 512*512
#define DATA_SIZE (NUMBER_OF_PARTICLES*4*sizeof(float)) 

cl_context mycontext;
cl_command_queue mycommandqueue;
cl_kernel mykernel;
cl_program myprogram;
cl_mem oclvbo, oclcbo, dev_velocity, dev_rseed;
size_t worksize[] = {NUMBER_OF_PARTICLES}; 
size_t lws[] = {128}; 

float host_position[NUMBER_OF_PARTICLES][4];
float host_velocity[NUMBER_OF_PARTICLES][4];
float host_rseed[NUMBER_OF_PARTICLES];
float host_color[NUMBER_OF_PARTICLES][4];

void do_kernel()
{
  cl_event waitlist[1];
 
  clEnqueueNDRangeKernel(mycommandqueue,mykernel,1,NULL,worksize,lws,0,0,
			 &waitlist[0]);
  clEnqueueReadBuffer(mycommandqueue,oclcbo,CL_TRUE,0,DATA_SIZE,host_color,0,NULL,NULL);
  clWaitForEvents(1,waitlist);
}

// void getPixelData()
// {

// }

void writeToPPM(int frame_num)
{ 
  int i,j;
  uint8_t *file_data;
  uint8_t *image_data;
  std::fstream ppm_file;
  std::stringstream sstm;
  std::string filename;
  sstm << "images/frame." << frame_num << ".ppm";
  filename = sstm.str();
  std::string header("P6\n1920\n1080\n255\n");
  std::string body("");
  image_data = (uint8_t*)calloc(WINDOW_HEIGHT*WINDOW_WIDTH*3,1);

  glReadPixels(0,0,WINDOW_WIDTH,WINDOW_HEIGHT,GL_RGB,GL_UNSIGNED_BYTE,image_data);
  for (i=0;i<WINDOW_WIDTH*WINDOW_HEIGHT*3;i++)
  {
    if ((i*2)%70==0) {
      body+="\n";
    }
    body+=*(image_data+i);
    body+=" ";
  }
  body += "\n";

  ppm_file.open(filename.c_str());
  ppm_file << header << body;
  ppm_file.close();
}

void mydisplayfunc()
{
  void *ptr;
  int count=0;
  glFinish();
  clEnqueueAcquireGLObjects(mycommandqueue,1,&oclvbo,0,0,0);
  do_kernel();
  clEnqueueReleaseGLObjects(mycommandqueue, 1, &oclvbo, 0,0,0);
  clFinish(mycommandqueue);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
  glVertexPointer(4,GL_FLOAT,0,0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER,0);
  glColorPointer(4,GL_FLOAT,0,&host_color);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glutSwapBuffers();
  glutPostRedisplay();
  writeToPPM(count++);
}

void setup_the_viewvol()
{
  float eye[] = {1.25, 1.9, 0.75};
  float view[] = {0.0, 0.3, 0.0};
  float up[] = {0.0, 1.0, 0.0};

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0,1.777,0.1,20.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}

void InitGL(int argc, char** argv)
{
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
  glutInitWindowSize(WINDOW_HEIGHT, WINDOW_WIDTH);
  glutInitWindowPosition(100,50);
  glutCreateWindow("my particle system");
  setup_the_viewvol();
  glPointSize(1.0);
  glLineWidth(3.0);
  glClearColor(0.0,0.0,0.0,1.0);
  glewInit();
  return;
}

double genrand()
{
  return(((double)(random()+1))/2147483649.);
}

void init_particles()
{
  int i, j;
  float color[4] = {0.917, 0.415, 0.125, 1.0};
  for(i=0;i<NUMBER_OF_PARTICLES;i++){
    host_position[i][0] = genrand()-0.5;
    host_position[i][1] = 0.2*genrand()+0.8;
    host_position[i][2] = genrand()-0.5;
    host_position[i][3] = 1.0;
    host_velocity[i][0] = 0.05*genrand();
    host_velocity[i][1] = 0.05*genrand();
    host_velocity[i][2] = 0.05*genrand();
    host_velocity[i][3] = 1.0;
    host_rseed[i] = genrand();
    for(j=0;j<4;j++)host_color[i][j] = color[j];
  }
}

void InitCL()
{
  int i;
  cl_platform_id myplatform;
  cl_device_id *mydevice;
  cl_int err;
  char* oclsource; 
  size_t program_length;
  unsigned int gpudevcount;

  err = RGUGetPlatformID(&myplatform);

  err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
  mydevice = new cl_device_id[gpudevcount];
  err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

  // You need all these to get full interoperability with OpenGL:
  cl_context_properties props[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
    0};

  mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
  mycommandqueue = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

  oclsource = RGULoadProgSource("particle_system.cl", "", &program_length);
  myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,
					&program_length, &err);
  if(err==CL_SUCCESS) fprintf(stderr,"create ok\n");
  else fprintf(stderr,"create err %d\n",err);
  clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
  mykernel = clCreateKernel(myprogram, "VVerlet", &err);
  if(err==CL_SUCCESS) fprintf(stderr,"build ok\n");
  else fprintf(stderr,"build err %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
  glBufferData(GL_ARRAY_BUFFER, DATA_SIZE, &host_position[0][0], GL_DYNAMIC_DRAW);
  oclvbo = clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);

  dev_velocity = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
				DATA_SIZE,&host_velocity[0][0],&err); 

  dev_rseed = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
			     NUMBER_OF_PARTICLES*sizeof(float),&host_rseed[0],&err);

  oclcbo = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
          DATA_SIZE,&host_color[0],&err);

  clSetKernelArg(mykernel,0,sizeof(cl_mem),(void *)&oclvbo);
  clSetKernelArg(mykernel,1,sizeof(cl_mem),(void *)&dev_velocity);
  clSetKernelArg(mykernel,2,sizeof(cl_mem),(void *)&dev_rseed);
  clSetKernelArg(mykernel,3,sizeof(cl_mem),(void *)&oclcbo);
}

void cleanup()
{
  clReleaseKernel(mykernel);
  clReleaseProgram(myprogram);
  clReleaseCommandQueue(mycommandqueue);
  glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
  glDeleteBuffers(1,&OGL_VBO);
  clReleaseMemObject(oclvbo);
  clReleaseMemObject(dev_velocity);
  clReleaseMemObject(dev_rseed);
  clReleaseMemObject(oclcbo);
  clReleaseContext(mycontext);
  exit(0);
}

void getout(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q':
    cleanup();
  default:
    break;
  }
}

int main(int argc,char **argv)
{
  srandom(123456789);
  init_particles();
  InitGL(argc, argv);
  InitCL(); 
  glutDisplayFunc(mydisplayfunc);
  glutKeyboardFunc(getout);
  glutMainLoop();
}
