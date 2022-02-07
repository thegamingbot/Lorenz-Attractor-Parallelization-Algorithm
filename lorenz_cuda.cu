/*
nvcc -o lorenz_cuda lorenz_cuda.cu -lGL -lGLU -lglut -lGLEW -lm
./lorenz_cuda
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536
#define NUM_BLOCKS 32
#define NUM_THREADS 2048

typedef struct point
{
    float x;
    float y;
    float z;
    float t;
} Point;

Point initial = {1.0, 1.0, 1.0, 0.0};

static GLfloat theta[] = {0.0, 0.0, 0.0};
GLint axis = 1;

__global__ void diff(Point *a)
{
    __shared__ float temp[4];
    __shared__ float sigma;
    __shared__ float rho;
    __shared__ float beta;
    __shared__ float dt;

    sigma = 10.0;
    rho = 28.0;
    beta = 8.0 / 3.0;
    dt = 0.01;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    switch (index)
    {
    case 0:
        temp[0] = sigma * (a->y - a->x);
        break;
    case 1:
        temp[1] = a->x * (rho - a->z) - a->y;
        break;
    case 2:
        temp[2] = a->x * a->y;
        break;
    case 3:
        temp[3] = beta * a->z;
        break;
    }

    // printf("%d\n", index);
    // printf("%f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);
    __syncthreads();

    if (index == 0)
    {
        printf("%f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);
        atomicAdd(&a->x, temp[0] * dt);
        atomicAdd(&a->y, temp[1] * dt);
        atomicAdd(&a->z, (temp[2] - temp[3]) * dt);
        atomicAdd(&a->t, dt);
    }
}

void draw(Point coord)
{
    glBegin(GL_POINTS);
    glColor3f(1, 1, 1);
    glVertex3f(coord.x, coord.y, coord.z);
    glEnd();
    glFlush();
    glutSwapBuffers();
}

void lorenzGenerator()
{
    Point *coord, *d_coord;
    coord = (Point *)malloc(sizeof(Point));
    cudaMalloc((void **)&d_coord, sizeof(Point));
    *coord = initial;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glPointSize(1.0);
    for (unsigned long int i = 0; i < N; i++)
    {
        cudaMemcpy(d_coord, coord, sizeof(Point), cudaMemcpyHostToDevice);
        diff<<<NUM_BLOCKS, NUM_THREADS>>>(d_coord);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

        cudaMemcpy(coord, d_coord, sizeof(Point), cudaMemcpyDeviceToHost);
        draw(*coord);
        break;
    }
    glutLeaveMainLoop();
    free(coord);
    cudaFree(d_coord);
}

void myReshape(int w, int h)
{
    glRotatef(theta[0], 1.0, 0.0, 0.0);
    glRotatef(theta[1], 0.0, 1.0, 0.0);
    glRotatef(theta[2], 0.0, 0.0, 1.0);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-50.0, 50.0, -50.0, 50.0, -50.0, 50.0);
    glMatrixMode(GL_MODELVIEW);
}

void spinCube()
{
    if (theta[axis] > 360.0)
        theta[axis] -= 360.0;
    else if (theta[axis] < 0)
        theta[axis] += 360.0;
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Lorenz");

    glutReshapeFunc(myReshape);
    glutIdleFunc(spinCube);
    glutDisplayFunc(lorenzGenerator);
    glEnable(GL_DEPTH_TEST);
    glutMainLoop();
}
