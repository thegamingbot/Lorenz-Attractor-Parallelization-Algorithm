/*
gcc -fopenmp -o lorenz_openmp lorenz_openmp.c -lGL -lGLU -lglut -lGLEW -lm
./lorenz_openmp
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <omp.h>

typedef struct point
{
    double x;
    double y;
    double z;
    float t;
} Point;

double sigma = 10.0;
double rho = 28.0;
double beta = 8.0 / 3.0;

Point initial = {1.0, 1.0, 1.0, 0.0};
double dt = 0.01;

unsigned long int start = 0;
unsigned long int end = 100000;

static GLfloat theta[] = {0.0, 0.0, 0.0};
GLint axis = 1;

unsigned long int op = 0;

Point differential(Point curr)
{
    Point diff;
    double temp0, temp1, temp2, temp3;
#pragma omp parallel sections
    {
#pragma omp section
        temp0 = sigma * (curr.y - curr.x);
#pragma omp section
        temp1 = curr.x * (rho - curr.z) - curr.y;
#pragma omp section
        temp2 = curr.x * curr.y;
#pragma omp section
        temp3 = beta * curr.z;
#pragma omp section
        diff.t = curr.t + dt;
    }
#pragma omp barrier
    {
        diff.x = temp0 * dt + curr.x;
        diff.y = temp1 * dt + curr.y;
        diff.z = (temp2 - temp3) * dt + curr.z;
    }
    return diff;
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
    Point coord = initial;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    double startTime = omp_get_wtime();
    glLoadIdentity();
    glPointSize(1.0);
    for (unsigned long int i = start; i < end; i++)
    {
        coord = differential(coord);
        draw(coord);
    }
    double endTime = omp_get_wtime();
    printf("%f\n", endTime - startTime);
    glutLeaveMainLoop();
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
    if (argc == 2)
        omp_set_num_threads(atoi(argv[1]));
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
