/*
mpicc -o lorenz_mpi lorenz_mpi.c -lGL -lGLU -lglut -lGLEW -lm
mpirun -n 5 ./lorenz_mpi
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <mpi.h>
#include <stddef.h>

typedef struct point
{
    double x;
    double y;
    double z;
    double t;
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

Point differential(Point curr)
{
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;

    MPI_Bcast(&curr, sizeof(Point), MPI_CHAR, 0, MPI_COMM_WORLD);

    Point diff;
    double temp0, temp1, temp2, temp3;
    switch (myid)
    {
    case 1:
        temp0 = sigma * (curr.y - curr.x);
        MPI_Send(&temp0, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        break;
    case 2:
        temp1 = curr.x * (rho - curr.z) - curr.y;
        MPI_Send(&temp1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        break;
    case 3:
        temp2 = curr.x * curr.y;
        MPI_Send(&temp2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        break;
    case 4:
        temp3 = beta * curr.z;
        MPI_Send(&temp3, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        MPI_Recv(&temp0, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&temp1, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&temp2, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&temp3, 1, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, &status);

        diff.x = temp0 * dt + curr.x;
        diff.y = temp1 * dt + curr.y;
        diff.z = (temp2 - temp3) * dt + curr.z;
        diff.t = curr.t + dt;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&diff, sizeof(Point), MPI_CHAR, 0, MPI_COMM_WORLD);
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
    glLoadIdentity();
    glPointSize(1.0);

    double startTime = MPI_Wtime();
    for (unsigned long int i = start; i < end; i++)
    {
        int myid, numprocs;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        coord = differential(coord);
        if (myid == 0)
            draw(coord);
    }
    double endTime = MPI_Wtime();
    printf("%f\n", endTime - startTime);

    MPI_Finalize();
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
    MPI_Init(NULL, NULL);

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
/*
1: 10.963762
6: 14.912688
8: 14.978625
12: 15.737066
16: 37.273723
20: 369.544086
24: 457.151230
32: 636.746772
48: 907.864457
64: 1298.230606
128: 1847.237682
*/
