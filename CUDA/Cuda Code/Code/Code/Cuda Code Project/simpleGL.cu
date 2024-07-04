/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* This code is a hevily modified form of the sampe code from the git hub repository https://github.com/NVIDIA/cuda-samples/tree/master
* This orignal sample can be found in Samples/5_Domain_Specific/simpleGL
* Almost all linker settings and include calls comes from and where set up with that
* And all functions orignally from the sample code has been either removed or heavily modified, and functions that have been slightly
* have been noted with a comment above the function to say its from simpleGL
* If a function has no comment abouve it, then it has either been completly recoded, or has been added out of source
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <curand_kernel.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>   

// CUDA helper functions
#include <helper_cuda.h>        

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.30f
#define REFRESH_DELAY 10

const unsigned int windowWidth  = 810;
const unsigned int windowHeight = 500;

const unsigned int N = 20;
const unsigned int M = 20;

GLuint vbo;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

//My values
const float squareSize = 0.01f;
const float gapSize = 0.08571428f;

const float initialMass = 0.0004f;
const float initialDampingCoeff = 0.003f;
const float springCoeff = 4.0f;
const float springRelaxDistance = 0.0956f;
const float distanceThreshold = 0.1f;

int runCount = 0;
std::vector<int> secondsToDisplay;

using clock2 = std::chrono::high_resolution_clock;

// Declare global variables
std::chrono::time_point<clock2> startTime;
std::chrono::time_point<clock2> endTime;
std::chrono::duration<double> deltaTime;

std::chrono::time_point<clock2> testTimer;

bool gravityEnabled = false;
bool visualsEnabled = false;
bool windEnabled = false;
bool testEnabled = false;

struct Point {
    float2 position;
    float2 prevPosition;
    float2 velocity;
    float2 externalForce;
    float mass;
    float dampingCoeff;
    int adjPoints[4];
    bool hasPhysics;
};

std::vector<Point> points;
std::vector<float> vertices;
std::vector<std::pair<int, int>> connections;

//End of my values

// mouse controls
int mouseOldX, mouseOldY;
int mouseButtons = 0;
float rotateX = 0.0, rotateY = 0.0;
float translateZ = -3.0;

StopWatchInterface *timer = NULL;

int fpsCount = 0;        
int fpsLimit = 1;        
int gIndex = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int gTotalErrors = 0;
bool gbQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

bool runSoftbody(int argc, char **argv, char *ref_file);
void cleanup();

bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo);
void deleteVBO(GLuint *vbo);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void runCuda();

void updateVBO();
void drawConnections();
void setupVertices();
void setupPoints();

const char *sSDKsample = "Cuda Softbody Simulation";

__global__ void updatePositions(Point* points, int width, int height, bool gravityEnabled, float springCoeff, float springRelaxDistance,float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if (idx < width && idy < height)
    {
        Point& p = points[index];

        if (p.hasPhysics == true)
        {
            float forceX = 0.0f;
            float forceY = 0.0f;


            for (int j = 0; j < 4; ++j)
            {
                int adj_index = p.adjPoints[j];
                if (adj_index != -1)
                {
                    Point& adj_point = points[adj_index];

                    float dx = adj_point.position.x - p.position.x;
                    float dy = adj_point.position.y - p.position.y;
                    float distance = sqrtf((dx * dx) + (dy * dy));
                    float magnitude = springCoeff * (distance - springRelaxDistance);
                    forceX += magnitude * dx / distance;
                    forceY += magnitude * dy / distance;

                }
            }


            forceX += -p.velocity.x * p.dampingCoeff;
            forceY += -p.velocity.y * p.dampingCoeff;

            if (gravityEnabled)
            {
                forceY += -9.81f * p.mass;
            }

            curandState_t state;
            curand_init(1234, idx, 0, &state);

            float randomNumber = 0.0f;

            if (p.externalForce.x != 0.0)
            {

                randomNumber = curand_uniform(&state);
                //randomNumber = randomNumber * 2.0f - 1.0f;

                forceX += randomNumber * p.externalForce.x;
            }

            if (p.externalForce.y != 0.0)
            {
                randomNumber = curand_uniform(&state);
                //randomNumber = randomNumber * 2.0f - 1.0f;

                forceY += randomNumber * p.externalForce.y;
            }

            float accelX = forceX / p.mass;
            float accelY = forceY / p.mass;

            float newPosX = p.position.x + p.velocity.x * deltaTime + 0.5f * accelX * (deltaTime * deltaTime);
            float newPosY = p.position.y + p.velocity.y * deltaTime + 0.5f * accelY * (deltaTime * deltaTime);

            __syncthreads();

            p.prevPosition = p.position;

            p.velocity.x = (newPosX - p.prevPosition.x) / deltaTime;
            p.velocity.y = (newPosY - p.prevPosition.y) / deltaTime;

            p.position.x = newPosX;
            p.position.y = newPosY;

            p.externalForce.x = 0;
            p.externalForce.y = 0;
        }
    }
}

// Main has barely been chaanged from simpleGL
int main(int argc, char **argv)
{
    startTime = clock2::now();
    testTimer = clock2::now();

    for (int i = 0; i <= 300; i += 10) {
        secondsToDisplay.push_back(i);
    }

    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    printf("\n");

    runSoftbody(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (gTotalErrors == 0) ? "OK" : "ERROR!");
    exit(gTotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

// computeFPS has barely been chaanged from simpleGL
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda Softbody Simulation: %3.1f fps", avgFPS);
    glutSetWindowTitle(fps);
}

// computeFPS has barely been chaanged from simpleGL however some extra function calls have been added
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Cuda Softbody Simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, windowWidth, windowHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)windowWidth / (GLfloat) windowHeight, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

// This has been modified to remove a set of unused code form orignal sample and has been renamed, oringal name was runCudaTests
bool runSoftbody(int argc, char **argv, char *ref_file)
{
    sdkCreateTimer(&timer);

    int devID = findCudaDevice(argc, (const char **)argv);

    if (ref_file != NULL)
    {

    }
    else
    {
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutCloseFunc(cleanup);


        setupPoints();

        createVBO(&vbo);

        runCuda();

        glutMainLoop();
    }

    return true;
}

void runCuda()
{
	for (int i = 1; i <= 10; ++i) {

		size_t num_bytes;
		Point* dPoints;
		checkCudaErrors(cudaMalloc(&dPoints, points.size() * sizeof(Point)));
		checkCudaErrors(cudaMemcpy(dPoints, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice));

		dim3 block(12, 12);
		dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);


		endTime = clock2::now();

		std::chrono::duration<double> deltaTime = endTime - startTime;
		float timePassed = deltaTime.count();

		if (timePassed > 0.06f)
		{
			timePassed = 0.01f;
		}
		updatePositions << <grid, block >> > (dPoints, N, M, gravityEnabled, springCoeff, springRelaxDistance, timePassed);

		startTime = clock2::now();

		checkCudaErrors(cudaMemcpy(points.data(), dPoints, points.size() * sizeof(Point), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(dPoints));

        if (testEnabled)
        {
            std::chrono::duration<double> time = clock2::now() - testTimer;
            runCount++;

            std::cout << "ACPS :" << runCount / time.count() << std::endl;
        }
	}
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

//This function has been untouched from source code
void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

void createVBO(GLuint* vbo)
{
    assert(vbo);

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    setupVertices();

    unsigned int numVertices = N * M * 6;
    unsigned int size = numVertices * 4 * sizeof(float);

    glBufferData(GL_ARRAY_BUFFER, size, vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    SDK_CHECK_ERROR_GL();
}



void updateVBO()
{
    int vertIndex = 0;
    for (unsigned int i = 0; i < M; ++i)
    {
        for (unsigned int j = 0; j < N; ++j)
        {
            float x0 = points[i * N + j].position.x;
            float y0 = points[i * N + j].position.y;
            float x1 = x0 + squareSize;
            float y1 = y0 + squareSize;

            vertices[vertIndex++] = x0; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x0; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;

            vertices[vertIndex++] = x0; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void setupVertices()
{
    unsigned int numVertices = N * M * 6;
    vertices.resize(numVertices * 4);

    float totalGapWidth = 2.0f - (N * squareSize);
    float totalGapHeight = 2.0f - (M * squareSize);
    float gapX = totalGapWidth / (N + 1);
    float gapY = totalGapHeight / (M + 1);

    int vertIndex = 0;

    for (unsigned int i = 0; i < M; ++i)
    {
        for (unsigned int j = 0; j < N; ++j)
        {
            float x0 = -1.0f + gapX * (j + 1) + squareSize * j;
            float y0 = -0.5f + gapY * (i + 1) + squareSize * i;

            float x1 = x0 + squareSize;
            float y1 = y0 + squareSize;

            vertices[vertIndex++] = x0; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x0; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;

            vertices[vertIndex++] = x0; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y0; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
            vertices[vertIndex++] = x1; vertices[vertIndex++] = y1; vertices[vertIndex++] = 0.0f; vertices[vertIndex++] = 1.0f;
        }
    }
}

void drawConnections()
{
    glColor3f(1.0f, 0.0f, 0.0f);

    glBegin(GL_QUADS);
    for (const auto& connection : connections)
    {
        const Point& p1 = points[connection.first];
        const Point& p2 = points[connection.second];

        float dx = p2.position.x - p1.position.x;
        float dy = p2.position.y - p1.position.y;
        float distance = sqrt(dx * dx + dy * dy);

        float unitX = dx / distance;
        float unitY = dy / distance;

        float perpX = -unitY;
        float perpY = unitX;

        float offsetX = perpX * squareSize / 2;
        float offsetY = perpY * squareSize / 2;

        float x1 = p1.position.x - offsetX;
        float y1 = p1.position.y - offsetY;
        float x2 = p1.position.x + offsetX;
        float y2 = p1.position.y + offsetY;
        float x3 = p2.position.x + offsetX;
        float y3 = p2.position.y + offsetY;
        float x4 = p2.position.x - offsetX;
        float y4 = p2.position.y - offsetY;

        glVertex3f(x1, y1, 0.0f);
        glVertex3f(x2, y2, 0.0f);
        glVertex3f(x3, y3, 0.0f);
        glVertex3f(x4, y4, 0.0f);
    }
    glEnd();
}

void setupPoints()
{
    points.resize(N * M);

    //float totalGapWidth = 2.0f - (N * squareSize);
    //float totalGapHeight = 2.0f - (M * squareSize);
    //float gapX = totalGapWidth / (N + 1);
    //float gapY = totalGapHeight / (M + 1);
    float gapY = gapSize;
    float gapX = gapSize;

    for (unsigned int i = 0; i < M; ++i)
    {
        for (unsigned int j = 0; j < N; ++j)
        {
            float x0 = -1.0f + gapX * (j + 1) + squareSize * j;
            float y0 = -0.5f + gapY * (i + 1) + squareSize * i;

            Point& p = points[i * N + j];
            p.position = make_float2(x0, y0);
            p.prevPosition = p.position;
            p.velocity = make_float2(0.0f, 0.0f);
            p.externalForce = make_float2(0.0f, 0.0f);
            p.mass = initialMass;
            p.dampingCoeff = initialDampingCoeff;
            p.hasPhysics = true;

            if ((i == M - 1 && j == 0) || (i == M - 1 && j == N - 1))
                p.hasPhysics = false;
        }
    }

    for (unsigned int i = 0; i < M; ++i)
    {
        for (unsigned int j = 0; j < N; ++j)
        {
            Point& p = points[i * N + j];
            int adjIndex = 0;
            if (i > 0)
            {
                p.adjPoints[adjIndex++] = (i - 1) * N + j;
            }
            if (i < M - 1)
            {
                p.adjPoints[adjIndex++] = (i + 1) * N + j;
            }
            if (j > 0)
            {
                p.adjPoints[adjIndex++] = i * N + (j - 1);
            }
            if (j < N - 1)
            {
                p.adjPoints[adjIndex++] = i * N + (j + 1);
            }
            for (; adjIndex < 4; ++adjIndex)
            {
                p.adjPoints[adjIndex] = -1;
            }

            for (int k = 0; k < 4; ++k)
            {
                if (p.adjPoints[k] != -1)
                {
                    int adjPointIndex = p.adjPoints[k];
                    if (i * N + j < adjPointIndex)
                    {
                        connections.emplace_back(i * N + j, adjPointIndex);
                    }
                }
            }
        }
    }
}


//This function had code removed that was no longer being used
void deleteVBO(GLuint *vbo)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

//Disply has been added too with extra function calls for new uses
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    if (windEnabled)
    {
        for (auto& point : points)
        {
            point.externalForce.x = 0.03f;
        }
    }

    runCuda();
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(visualsEnabled)
    {
    updateVBO();

    

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translateZ);
    glRotatef(rotateX, 1.0, 0.0, 0.0);
    glRotatef(rotateY, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, (N * M) * 6);
    glDisableClientState(GL_VERTEX_ARRAY);

    drawConnections();
    }
    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

// This function has been untoched since source
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

// This function has been untoched since source
void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo);
    }
}

// This function has been added appon for more functions
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case 'g':
    case 'G':
        gravityEnabled = !gravityEnabled;
        break;
    case 'v':
    case 'V':
        visualsEnabled = !visualsEnabled;
        break;
    case 'w':
    case 'W':
        windEnabled = !windEnabled;
        break;
    case 27:
        glutDestroyWindow(glutGetWindow());
        return;
    }
}

float distance(float2 a, float2 b) {
    return sqrtf((a.x - b.x) * (a.y - b.y) + (a.y - b.y) * (a.y - b.y));
}

float2 normalize(float2 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y);
    if (length != 0) {
        v.x /= length;
        v.y /= length;
    }
    return v;
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouseButtons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouseButtons = 0;
    }

    mouseOldX = x;
    mouseOldY = y;
}

void motion(int x, int y)
{
    float worldX = (x / (float)windowWidth) * 2.0f - 1.0f;
    float worldY = 1.0f - (y / (float)windowHeight) * 2.0f;

    float2 mousePos = make_float2(worldX, worldY);

    for (auto& point : points)
    {
        float dist = distance(mousePos, point.position);
        if (dist < distanceThreshold)
        {
            float2 mouse_direction = make_float2(mousePos.x - point.position.x, mousePos.y - point.position.y);
            mouse_direction = normalize(mouse_direction);
            point.externalForce = make_float2(mouse_direction.x * 5.0f, mouse_direction.y * 5.0f);
        }
    }
}
