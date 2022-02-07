/*
gcc -fopenmp -o lorenz_serial lorenz_serial.c -lGL -lGLU -lglut -lGLEW -lm
./lorenz_serial
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
unsigned long int end = 4096;

static GLfloat theta[] = {0.0, 0.0, 0.0};
GLint axis = 1;

unsigned long int op = 0;

Point differential(Point curr)
{
    Point diff;
    // 4 artih ops, 5 reads, 1 write
    diff.x = (sigma * (curr.y - curr.x)) * dt + curr.x;
    op += 4;
    // 5 arith ops, 6 reads, 1 write
    diff.y = (curr.x * (rho - curr.z) - curr.y) * dt + curr.y;
    op += 5;
    // 5 artih ops, 6 reads, 1 write
    diff.z = (curr.x * curr.y - beta * curr.z) * dt + curr.z;
    op += 5;
    // 1 arith op, 2 reads, 1 write
    diff.t = curr.t + dt;
    op += 1;
    return diff;
}

void lorenzGenerator()
{
    Point coord = initial;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glRotatef(theta[0], 1.0, 0.0, 0.0);
    glRotatef(theta[1], 0.0, 1.0, 0.0);
    glRotatef(theta[2], 0.0, 0.0, 1.0);
    glPointSize(1.0);
    FILE *fp = fopen("new.txt", "w");
    double startTime = omp_get_wtime();
    for (unsigned long int i = start; i < end; i++)
    {
        coord = differential(coord);
        fprintf(fp, "%f, %f, %f %f\n", coord.t, coord.x, coord.y, coord.z);
        glBegin(GL_POINTS);
        glColor3f(1, 1, 1);
        glVertex3f(coord.x, coord.y, coord.z);
        glEnd();
        glFlush();
        glutSwapBuffers();
    }
    fclose(fp);
    double endTime = omp_get_wtime();
    printf("%f\n", endTime - startTime);
    glutLeaveMainLoop();
}

void spinCube()
{
    if (theta[axis] > 360.0)
        theta[axis] -= 360.0;
    else if (theta[axis] < 0)
        theta[axis] += 360.0;
    glutPostRedisplay();
}

void myReshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-50.0, 50.0, -50.0, 50.0, -50.0, 50.0);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1280, 720);

    glutCreateWindow("Lorenz");
    glutReshapeFunc(myReshape);
    glutDisplayFunc(lorenzGenerator);
    glutIdleFunc(spinCube);
    glEnable(GL_DEPTH_TEST);
    glutMainLoop();
}
