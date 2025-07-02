// --------------------------------
// INCLUDES
// --------------------------------

// base
#include "sph.h"

// sph
#include "particle.h"

// Qt
#include <QElapsedTimer>

// cmath
#include <math.h>

// openmp
#include <omp.h>

#include <QDateTime>

// write
#include <iostream>
#include <fstream>
#include <sys/stat.h> 
#include <sys/types.h> // write
#include <iostream>
#include <fstream>
#include <sys/stat.h>
// --------------------------------


SPH::SPH()
 : mParticleCount(0),
   mGridCellCount(0),
   mRho0(0.0f),
   mStopped(false),
   mPaused(false),
   mKineticEnergyTotal(0.0f),
   mPotentialEnergyTotal(0.0f),
   mAngularMomentumTotal(vec3(0.0f, 0.0f, 0.0f))
{
   // Unidades: [km/s pc M_sun Myr]...
   // Init scales
   float h = 0.1f;  // Distancia para búsqueda de vecinos
   mParticleCount = 16 * 1024;  // Cantidad de partículas (~10k < N < ~100k)
   mExamineCount = 32;  // Cantidad target de vecinos (16 < K < 64)
   mGridCellsX = 32;  // Cantidad de celdas a lo largo de 1 eje (16 < M < 64)
   mGridCellsY = 32;
   mGridCellsZ = 32;
   mGridCellCount = mGridCellsX * mGridCellsY * mGridCellsZ;
   mCellSize = 2.0f * h;  // Tamaño de celda
   mMaxX = mCellSize * mGridCellsX;  // Bordes en cada eje
   mMaxY = mCellSize * mGridCellsY;
   mMaxZ = mCellSize * mGridCellsZ;
   // --------------------------------

   // Legacy & helpers ctes
   mSimulationScale = 1.0f;
   mSimulationScaleInverse = 1.0f / mSimulationScale;
   mH = h;
   mH2 = h * h;
   mHTimes2 = h * 2.0f;
   mHTimes2Inv = 1.0f / mHTimes2;
   mHScaled  = h * mSimulationScale;
   mHScaled2 = pow(h * mSimulationScale, 2);
   mHScaled6 = pow(h * mSimulationScale, 6);
   mHScaled9 = pow(h * mSimulationScale, 9);
   mCflLimit = 1e+4f;  // Safety limit for accel calculation
   mCflLimit2 = mCflLimit * mCflLimit;
   mKernel1Scaled = 315.0f / (64.0f * (float)(M_PI) * mHScaled9);  // Smoothing kernels
   mKernel2Scaled = -45.0f / ((float)(M_PI) * mHScaled6);
   mKernel3Scaled = -mKernel2Scaled;
   // --------------------------------

   // Tiempo
   float time_simu = 1.0f;  // Total
   mTimeStep = 1e-3f;  // dt
   totalSteps = (int)round(time_simu/mTimeStep);
   // --------------------------------

   // Parámetros físicos
   mRho0 = 100.0f;  // Target density
   mStiffness = 0.1f;  // Rigidez de cada smooth particle
   mGravity = vec3(0.0f, 0.0f, 0.0f);  // Gravedad uniforme de background
   mViscosityScalar = 100.0f;  // Viscosidad
   mDamping = 0.001f;  // Coeficiente de rebote si boundary is on.
   mGravConstant = 4.3009e-3f;  // Cte univ de gravedad en [pc (km/s)^2 / M_sun]
   mCentralMass = 1e+5f;  // Masa central
   mCentralPos = vec3(mMaxX * 0.5f, mMaxY * 0.5f, mMaxZ * 0.5f);  // Ubicación de la misma (centro)
   mSoftening = mHScaled * 0.5;  // Force-softening length (0.5*h < h < 2*h)
   float mass = 1.0f;  // Masa de cada partícula
   // --------------------------------


    mSrcParticles = new Particle[mParticleCount];
   mVoxelIds= new int[mParticleCount];
   mVoxelCoords= new vec3i[mParticleCount];

   // Para difs masas (estaria bueno ver que onda...)
   for (int i = 0; i < mParticleCount; i++)
   {
      mSrcParticles[i].mMass = mass;
   }

   mGrid = new QList<Particle*>[mGridCellCount];

   mNeighbors = new Particle*[mParticleCount*mExamineCount];
   mNeighborDistancesScaled = new float[mParticleCount*mExamineCount];

   // Chose initial configuration
   // initParticlePositionsRandom();  // Unif en los 3 ejes i.e. ~caja
   initParticlePolitionsSphere();  // Esfera con rotación inicial
}

SPH::~SPH()
{
   stopSimulation();
   quit();
   wait();
}


bool SPH::isStopped() const
{
   mMutex.lock();
   bool stopped = mStopped;
   mMutex.unlock();

   return stopped;
}


bool SPH::isPaused() const
{
   mMutex.lock();
   bool paused = mPaused;
   mMutex.unlock();

   return paused;
}



void SPH::run()
{
   int stepCount = 0;

   /*
   Just for debugging (temporal evolution of Energy and Angular momentum)

   // Create directory ./out
   const char *path = "out";
   int result = mkdir(path, 0777);
   if (result == 0)
      std::cout << "Directory created" << std::endl;
   else
      std::cout << "Directory already exists" << std::endl;

   // Create files
   std::ofstream outfile1("out/energy.txt");
   outfile1 << "Step, Kinetic Energy, Potential Energy, Total Energy" << std::endl;
   std::ofstream outfile2("out/angularmomentum.txt");
   outfile2 << "Step, Angular Momentum" << std::endl;
   std::ofstream outfile3("out/timing.txt");
   outfile3 << "Step, Voxelize, Find Neighbors, Compute Density, Compute Pressure, Compute Acceleration, Integrate" << std::endl;
   */

   // Main loop
   while(!isStopped() && stepCount <= totalSteps)
   {
      if (!isPaused())
      {
         step();

         // Write files (if debugging)
         //outfile1 << stepCount << ", " << mKineticEnergyTotal << ", " << mPotentialEnergyTotal << ", " << mKineticEnergyTotal + mPotentialEnergyTotal << std::endl;
         //outfile2 << stepCount << ", " << mAngularMomentumTotal.length() << std::endl;
         //outfile3 << stepCount << ", " << timeVoxelize << ", " << timeFindNeighbors << ", " << timeComputeDensity << ", " << timeComputePressure << ", " << timeComputeAcceleration << ", " << timeIntegrate << std::endl;

         stepCount++;
      }
   }

   // Close files (if debugging)
   //outfile1.close();
   //outfile2.close();
   //outfile3.close();
}


void SPH::step()
{
   // Init timers
   timeVoxelize = 0;
   timeFindNeighbors = 0;
   timeComputeDensity = 0;
   timeComputePressure = 0;
   timeComputeAcceleration = 0;
   timeIntegrate = 0;
   QElapsedTimer t;
   mKineticEnergyTotal = 0.0f;
   mPotentialEnergyTotal = 0.0f;
   mAngularMomentumTotal = vec3(0.0f, 0.0f, 0.0f);
   // --------------------------------

   // Put particles into a grid of voxels
   t.start();
   voxelizeParticles();
   timeVoxelize = t.nsecsElapsed() / 1000000;

   // Find neighbors for each particle
   t.start();
   for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
   {
      Particle* particle = &mSrcParticles[particleIndex];
      const vec3i& voxel= mVoxelCoords[particleIndex];

      // Neighbors for this particle
      Particle** neighbors= &mNeighbors[particleIndex*mExamineCount];

      // Examine a local region of 2x2x2 voxels using the spatial index to
      // look up for near particles.
      findNeighbors(particle, particleIndex, neighbors, voxel.x, voxel.y, voxel.z);
   }
   timeFindNeighbors = t.nsecsElapsed() / 1000000;

   // Compute density for each particle (using its negihbors)
   t.start();
   for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
   {
      Particle* particle = &mSrcParticles[particleIndex];

      // Neighbors for this particle
      Particle** neighbors= &mNeighbors[particleIndex*mExamineCount];
      float* neighborDistances= &mNeighborDistancesScaled[particleIndex*mExamineCount];

      computeDensity(particle, neighbors, neighborDistances);
   }
   timeComputeDensity = t.nsecsElapsed() / 1000000;

   // Compute pressure (1st part of accel)
   // Use this function ONLY IF AN EQUATION OF STATE IS WANTED
   // (Skip because can be done on-the-fly)
   t.start();
   /*
   for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
   {
      Particle* particle = &mSrcParticles[particleIndex];

      computePressure(particle);
   }
   */
   timeComputePressure = t.nsecsElapsed() / 1000000;

   // Compute acceleration (hydro + gravity)
   t.start();
   for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
   {
      Particle* particle = &mSrcParticles[particleIndex];

      // Neighbors for this particle
      Particle** neighbors= &mNeighbors[particleIndex*mExamineCount];
      float* neighborDistances= &mNeighborDistancesScaled[particleIndex*mExamineCount];

      computeAcceleration(particle, neighbors, neighborDistances);
   }
   timeComputeAcceleration = t.nsecsElapsed() / 1000000;

   // Kinematic integration -> LeapFrog KDK instead of an explicit Euler
   t.start();
   for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
   {
      Particle* particle = &mSrcParticles[particleIndex];

      integrate(particle);

      // Only if debugging -> Energy & Angular Momentum
      //mKineticEnergyTotal += 0.5f * particle->mMass * particle->mVelocity.length2();
      //mAngularMomentumTotal += (particle->mMass * \
         (particle->mPosition - mCentralPos).cross(particle->mVelocity));
   }
   timeIntegrate = t.nsecsElapsed() / 1000000;

   // Pass the timers to the GUI
   emit updateElapsed(
      timeVoxelize,
      timeFindNeighbors,
      timeComputeDensity,
      timeComputePressure,
      timeComputeAcceleration,
      timeIntegrate
   );

   emit stepFinished();
}


// GUI-wise
void SPH::pauseResume()
{
   mMutex.lock();
   mPaused = !mPaused;
   mMutex.unlock();
}


void SPH::stopSimulation()
{
   mMutex.lock();
   mStopped = true;
   mMutex.unlock();
}
// --------------------------------


// Box initial conditions (Legacy)
void SPH::initParticlePositionsRandom()
{
   // srand(QDateTime::currentMSecsSinceEpoch() % 1000);

   // for (int i = 0; i < mParticleCount; i++)
   // {
   //    float x = rand() / (float)RAND_MAX;
   //    float y = rand() / (float)RAND_MAX;
   //    float z = rand() / (float)RAND_MAX;

   //    x *= mGridCellsX * mHTimes2 * 0.1f;
   //    y *= mGridCellsY * mHTimes2 * 0.75f;
   //    z *= mGridCellsZ * mHTimes2;

   //    if (x == (float)mGridCellsX)
   //       x -= 0.00001f;
   //    if (y == (float)mGridCellsY)
   //       y -= 0.00001f;
   //    if (z == (float)mGridCellsZ)
   //       z -= 0.00001f;
   //    mSrcParticles->mPosition[i].set(x, y, z);
   // }

   // // just set up random directions
   // for (int i = 0; i < mParticleCount; i++)
   // {
      
   //    // have a range from -1 to 1
   //    float x = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
   //    float y = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
   //    float z = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;

   //    mSrcParticles->mVelocity[i].set(x, y, z);
   // }
}


// Disc initial conditions (fiducial)
void SPH::initParticlePolitionsSphere()
{
   // Fix seed:
   srand(42);

   float dist = 0.0f;

   float x = 0.0f;
   float y = 0.0f;
   float z = 0.0f;

   vec3 sphereCenter;
   sphereCenter.set(
      mMaxX * 0.5f,
      mMaxY * 0.5f,
      mMaxZ * 0.5f
   );

   float radius = 2.0f;
   float phi;  // El ang acimutal para la v_tangencial. (atan2(y,x))
   float v_x_inic, v_y_inic, v_z_inic;  // Y is the VERTICAL component

   for (int i = 0; i < mParticleCount; i++)
   {
      do
      {
         x = rand() / (float)RAND_MAX;
         y = rand() / (float)RAND_MAX;
         z = rand() / (float)RAND_MAX;

         x *= mGridCellsX * mHTimes2;
         y *= mGridCellsY * mHTimes2;
         z *= mGridCellsZ * mHTimes2;

         if (x == (float)mGridCellsX)
            x -= 0.00001f;
         if (y == (float)mGridCellsY)
            y -= 0.00001f;
         if (z == (float)mGridCellsZ)
            z -= 0.00001f;

         dist = (vec3(x,y,z) - sphereCenter).length();
      }
      while (dist > radius);

      mSrcParticles[i].mPosition.set(x, y, z);
      phi = atan2(z - mMaxZ * 0.5f, x - mMaxX * 0.5f);
      // Rotation (The factor 20 is arbitrary here)
      v_x_inic = 20.0f * pow(dist + mHScaled*0.5, -0.5) * -sin(phi);
      v_z_inic = 20.0f * pow(dist + mHScaled*0.5, -0.5) * cos(phi);
      // Some random movements on the vertical component (arbitrary constants).
      v_y_inic = dist * (((rand() / (float)RAND_MAX) * 2.0f) - 1.0f);
      // Append
      mSrcParticles[i].mVelocity.set(v_x_inic, v_y_inic, v_z_inic);
   }

}



void SPH::clearGrid()
{
   for (int i = 0; i < mGridCellCount; i++)
   {
      mGrid[i].clear();
   }
}


// Populate the voxels of the grid
void SPH::voxelizeParticles()
{
   clearGrid();

   for (int i = 0; i < mParticleCount; i++)
   {
      Particle* particle = &mSrcParticles[i];

      // Compute a scalar voxel id from a position
      vec3 pos = particle->mPosition;

      int voxelX = (int)floor(pos.x * mHTimes2Inv);
      int voxelY = (int)floor(pos.y * mHTimes2Inv);
      int voxelZ = (int)floor(pos.z * mHTimes2Inv);

      if (voxelX < 0) voxelX= 0;
      if (voxelY < 0) voxelY= 0;
      if (voxelZ < 0) voxelZ= 0;
      if (voxelX >= mGridCellsX) voxelX= mGridCellsX-1;
      if (voxelY >= mGridCellsY) voxelY= mGridCellsY-1;
      if (voxelZ >= mGridCellsZ) voxelZ= mGridCellsZ-1;

      mVoxelCoords[i].x= voxelX;
      mVoxelCoords[i].y= voxelY;
      mVoxelCoords[i].z= voxelZ;

      int voxelId = computeVoxelId(voxelX, voxelY, voxelZ);
      mVoxelIds[i]= voxelId;
   }

   // Put each particle into according voxel
   for (int i = 0; i < mParticleCount; i++)
   {
       Particle* particle = &mSrcParticles[i];
       mGrid[ mVoxelIds[i] ].push_back(particle);
   }
}


// Search for neighboring particles
void SPH::findNeighbors(Particle* particle, int particleIndex, Particle** neighbors, int voxelX, int voxelY, int voxelZ)
{
   float xOrientation = 0.0f;
   float yOrientation = 0.0f;
   float zOrientation = 0.0f;

   int x = 0;
   int y = 0;
   int z = 0;

   int particleOffset = 0;
   int particleIterateDirection = 0;
   int neighborIndex = 0;
   bool enoughNeighborsFound = false;

   vec3 pos = particle->mPosition;

   // The relative position; i.e the orientation within a voxel
   xOrientation = pos.x - (voxelX * mHTimes2);
   yOrientation = pos.y - (voxelY * mHTimes2);
   zOrientation = pos.z - (voxelZ * mHTimes2);

   // Neighbour voxels
   x = 0;
   y = 0;
   z = 0;

   // Location within voxel
   (xOrientation > mH) ? x++ : x--;
   (yOrientation > mH) ? y++ : y--;
   (zOrientation > mH) ? z++ : z--;

   int vx[8];
   int vy[8];
   int vz[8];

   vx[0] = voxelX;
   vy[0] = voxelY;
   vz[0] = voxelZ;

   vx[1] = voxelX + x;
   vy[1] = voxelY;
   vz[1] = voxelZ;

   vx[2] = voxelX;
   vy[2] = voxelY + y;
   vz[2] = voxelZ;

   vx[3] = voxelX;
   vy[3] = voxelY;
   vz[3] = voxelZ + z;

   vx[4] = voxelX + x;
   vy[4] = voxelY + y;
   vz[4] = voxelZ;

   vx[5] = voxelX + x;
   vy[5] = voxelY;
   vz[5] = voxelZ + z;

   vx[6] = voxelX;
   vy[6] = voxelY + y;
   vz[6] = voxelZ + z;

   vx[7] = voxelX + x;
   vy[7] = voxelY + y;
   vz[7] = voxelZ + z;

   int vxi;
   int vyi;
   int vzi;

   for (int voxelIndex = 0; voxelIndex < 8; voxelIndex++)
   {
      vxi = vx[voxelIndex];
      vyi = vy[voxelIndex];
      vzi = vz[voxelIndex];

      // Sanity-check
      if (
            vxi > 0 && vxi < mGridCellsX
         && vyi > 0 && vyi < mGridCellsY
         && vzi > 0 && vzi < mGridCellsZ
      )
      {
         const QList<Particle*>& voxel = mGrid[computeVoxelId(vxi, vyi, vzi)];

         if (!voxel.isEmpty())
         {
            // Random offset (we don't wanna take always the same nieghbors)
            particleOffset = rand() % voxel.length();
            particleIterateDirection = (particleIndex % 2) ? -1 : 1;

            int i = 0;
            while (true)
            {
               int nextIndex = particleOffset + i * particleIterateDirection;

               // Leave if we're out out the voxel's bounds
               if (nextIndex < 0 || nextIndex > voxel.length() - 1)
                  break;

               Particle* neighbor = voxel[nextIndex];
               i++;

               Particle* validNeighbor = evaluateNeighbor(particle, neighbor);

               if (validNeighbor)
               {
                  neighbors[neighborIndex] = validNeighbor;
                  neighborIndex++;
               }

               // Leave if we have sufficient neighbor particles
               enoughNeighborsFound = (neighborIndex > mExamineCount - 1);
               if (enoughNeighborsFound)
                  break;
            }
         }
      }

      // Leave if we have sufficient neighbor particles
      if (enoughNeighborsFound)
         break;
   }

   particle->mNeighborCount = neighborIndex;
}


// Neighbor evaluation (distance checker)
Particle* SPH::evaluateNeighbor(
   Particle* current,
   Particle* neighbor
)
{
   Particle* validNeighbor = 0;

   if (current != neighbor)
   {
      vec3 dist = current->mPosition - neighbor->mPosition;
      float dot = dist * dist;

      // If close => Valid
      if (dot < mH2)
      {
         validNeighbor = neighbor;
      }
   }

   return validNeighbor;
}


// Individual particle density (using its neighbors)
void SPH::computeDensity(Particle* particle, Particle** neighbors, float* neighborDistances)
{
   float density = 0.0f;
   float mass = 0.0f;
   vec3 pos = particle->mPosition;
   float w = 0.0f;
   float rightPart = 0.0f;
   float distanceScaled;

   // Considering the already checked neighbors
   for (int neighborIndex = 0; neighborIndex < particle->mNeighborCount; neighborIndex++)
   {
      Particle* neighbor = neighbors[neighborIndex];

      if (!neighbor)
         break;

      if (neighbor != particle)
      {
         // Mass of neighbor
         mass = neighbor->mMass;

         // Smoothing kernel         
         vec3 dist = pos - neighbor->mPosition;
         float dot = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
         float distance = sqrt(dot);
         float distanceScaled = distance * mSimulationScale;
         neighborDistances[neighborIndex] = distanceScaled;

         if (distanceScaled > mHScaled)
         {
            w = 0.0f;
         }
         else
         {
            rightPart = (mHScaled2 - (distanceScaled * distanceScaled));
            rightPart = (rightPart * rightPart * rightPart);
            w = mKernel1Scaled * rightPart;

            // Weighted neighbor mass to density
            density += (mass * w);
         }
      }
   }

   particle->mDensity = density;
}

// Calc the pressure on-the-fly, maybe change later (diff EoS)
void SPH::computePressure(Particle* particle)
{
   // rho0: resting density
   // float deltaRho = particle->mDensity - mRho0;
   // float p = mStiffness * deltaRho;
   // particle->mPressure = p;
}


// Dynamics of each particle (hydro + grav)
void SPH::computeAcceleration(Particle* p, Particle** neighbors, float* neighborDistances)
{
   // HYDRO:
   // Init vars (this particle)
   Particle* neighbor = 0;
   float distanceToNeighborScaled = 0.0f;
   float pi = (p->mDensity - mRho0) * mStiffness;
   float rhoiInv = ((p->mDensity > 0.0f) ? (1.0f / p->mDensity) : 1.0f);
   float rhoiInv2 = rhoiInv * rhoiInv;
   float piDivRhoi2 = pi * rhoiInv2;
   vec3 r = p->mPosition;
   vec3 vi = p->mVelocity;
   // Init vars (for neighbors)
   float pj = 0.0f;
   float rhoj = 0.0f;
   float rhojInv = 0.0f;
   float rhojInv2 = 0.0f;
   float mj = 0.0f;
   vec3 rj;
   vec3 vj;
   vec3 rMinusRj;
   vec3 rMinusRjScaled;

   // Pressure gradient
   vec3 pressureGradient(0.0f, 0.0f, 0.0f);
   vec3 pressureGradientContribution;
   // Viscous term
   vec3 viscousTerm(0.0f, 0.0f, 0.0f);
   vec3 viscousTermContribution;
   // Total acceleration
   vec3 acceleration(0.0f, 0.0f, 0.0f);

   // Sweep through neighbors
   for (int neighborIndex = 0; neighborIndex < p->mNeighborCount; neighborIndex++)
   {
      neighbor = neighbors[neighborIndex];

      pj = (neighbor->mDensity - mRho0) * mStiffness;
      rhoj = neighbor->mDensity;
      rhojInv = ((rhoj > 0.0f) ? (1.0f / rhoj) : 1.0f);
      rhojInv2 = rhojInv * rhojInv;
      rj = neighbor->mPosition;
      vj = neighbor->mVelocity;
      mj = neighbor->mMass;

      // Pressure gradient
      rMinusRj = (r - rj);
      rMinusRjScaled = rMinusRj * mSimulationScale;
      distanceToNeighborScaled = neighborDistances[neighborIndex];
      pressureGradientContribution = rMinusRjScaled / distanceToNeighborScaled;
      pressureGradientContribution *= mKernel2Scaled;

      float centerPart = (mHScaled - distanceToNeighborScaled);
      centerPart = centerPart * centerPart;
      pressureGradientContribution *= centerPart;

      float factor = mj * piDivRhoi2 * (pj * rhojInv2);
      pressureGradientContribution *= factor;

      // Add individual pressure gradient contribution to total pressure gradient
      pressureGradient += pressureGradientContribution;

      // Viscosity
      viscousTermContribution = vj - vi;
      viscousTermContribution *= rhojInv;
      viscousTermContribution *= mj;

      viscousTermContribution *= mKernel3Scaled;
      viscousTermContribution *= (mHScaled - distanceToNeighborScaled);

      // Add individual contribution to total viscous term
      viscousTerm += viscousTermContribution;
   }

   viscousTerm *= (mViscosityScalar * rhoiInv);

   // Acceleration by hydrodynamics
   acceleration = viscousTerm - pressureGradient;

   // Gravity
   vec3 gravityTerm(0.0f, 0.0f, 0.0f);
   vec3 gravityTermContribution;
   float distance_ij3;
   // Central mass
   rMinusRj = (r - mCentralPos);
   rMinusRjScaled = rMinusRj * mSimulationScale;
   distance_ij3 = pow((rMinusRjScaled.length() + mSoftening), 3);
   gravityTermContribution = rMinusRjScaled/distance_ij3;
   gravityTermContribution *= -mGravConstant * mCentralMass;
   gravityTerm += gravityTermContribution;

   // Update accel with gravity
   acceleration += gravityTerm;

   // Sanity-check for the CFL condition
   float dot =
        acceleration.x * acceleration.x
      + acceleration.y * acceleration.y
      + acceleration.z * acceleration.z;

   bool limitExceeded = (dot > mCflLimit2);
   if (limitExceeded)
   {
      float length = sqrt(dot);
      float cflScale = mCflLimit / length;
      acceleration *= cflScale;
   }

   // If debug, potential energy:
   //mPotentialEnergyTotal += -mGravConstant * mCentralMass * p->mMass / (rMinusRjScaled.length());

   // Append accel
   p->mAcceleration = acceleration;
}


// Kinemtaics integrator
void SPH::integrate(Particle* p)
{
   vec3 acceleration = p->mAcceleration;
   vec3 position = p->mPosition;
   vec3 velocity = p->mVelocity;

   float posTimeStep = mTimeStep * mSimulationScaleInverse;

   // LF-KDK:
   
   // We need to recompute the accel in the half-step (or use that of the previous step)
   vec3 velocity_halfstep;
   velocity_halfstep = velocity + (acceleration * mTimeStep*0.5);
   vec3 newPosition = position + (velocity_halfstep * posTimeStep);
   vec3 newAcceleration;

   float distance_ij3;
   vec3 rMinusRj  = (newPosition - mCentralPos);
   vec3 rMinusRjScaled = rMinusRj * mSimulationScale;
   distance_ij3 = pow((rMinusRjScaled.length() + mSoftening), 3);
   newAcceleration = rMinusRjScaled/distance_ij3;
   newAcceleration *= -mGravConstant * mCentralMass;

   vec3 newVelocity = velocity_halfstep + (newAcceleration * mTimeStep);

   // If wanted, check boundaries (if not, let them escape the system)
   // handleBoundaryConditions(
   //    position,
   //    &newVelocity,
   //    posTimeStep,
   //    &newPosition
   // );

   p->mVelocity = newVelocity;
   p->mPosition = newPosition;

   // If debugging, update the energy:
   //p->mKineticEnergy = 0.5f * p->mMass * (newVelocity.x * newVelocity.x + newVelocity.y * newVelocity.y + newVelocity.z * newVelocity.z);
   // Angular momentum m * (r x v) ; w.r.t. to central mass (not origin)
   //p->mAngularMomentum = p->mMass * (rMinusRj.cross(newVelocity));
   // It's negative because we are using "y" as the vertical; (x, z) => ^x X ^z = -^y...
}


// --------------------------------
// HELPERS & GUI
// --------------------------------
void SPH::handleBoundaryConditions(
   vec3 position,
   vec3* newVelocity,
   float timeStep,
   vec3* newPosition
)
{
   // x coord
   if (newPosition->x < 0.0f)
   {
      vec3 normal(1, 0, 0);
      float intersectionDistance = -position.x / newVelocity->x;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->x > mMaxX)
   {
      vec3 normal(-1, 0, 0);
      float intersectionDistance = (mMaxX - position.x) / newVelocity->x;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }

   // y coord
   if (newPosition->y < 0.0f)
   {
      vec3 normal(0, 1, 0);
      float intersectionDistance = -position.y / newVelocity->y;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->y > mMaxY)
   {
      vec3 normal(0, -1, 0);
      float intersectionDistance = (mMaxY - position.y) / newVelocity->y;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }

   // z coord
   if (newPosition->z < 0.0f)
   {
      vec3 normal(0, 0, 1);
      float intersectionDistance = -position.z / newVelocity->z;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->z > mMaxZ)
   {
      vec3 normal(0, 0, -1);
      float intersectionDistance = (mMaxZ - position.z) / newVelocity->z;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
}


void SPH::applyBoundary(
      vec3 position,
      float timeStep,
      vec3* newPosition,
      float intersectionDistance,
   vec3 normal,
   vec3* newVelocity
)
{
   
   vec3 intersection = position + (*newVelocity * intersectionDistance);

   float dotProduct =
        newVelocity->x * normal.x
      + newVelocity->y * normal.y
      + newVelocity->z * normal.z;

   vec3 reflection = *newVelocity - (normal * dotProduct * 2.0f);

   float remaining = timeStep - intersectionDistance;

   // apply boundaries
   *newVelocity = reflection;
   *newPosition = intersection + reflection * (remaining * mDamping);
}


int SPH::computeVoxelId(int voxelX, int voxelY, int voxelZ)
{
   return (voxelZ * mGridCellsY + voxelY) * mGridCellsX + voxelX;
}


void SPH::clearNeighbors()
{
   memClear32(mNeighbors, mParticleCount * mExamineCount * sizeof(Particle*));
}


void SPH::memClear32(void* dst, int len)
{
   unsigned int* dst32= (unsigned int*)dst;
   len>>=2;
   while (len--)
      *dst32++= 0;
}


float SPH::getCellSize() const
{
   return mCellSize;
}


Particle* SPH::getParticles()
{
   return mSrcParticles;
}


int SPH::getParticleCount() const
{
   return mParticleCount;
}


void SPH::getGridCellCounts(int &x, int &y, int &z)
{
   x = mGridCellsX;
   y = mGridCellsY;
   z = mGridCellsZ;
}


void SPH::getParticleBounds(float &x, float &y, float &z)
{
   x = mMaxX;
   y = mMaxY;
   z = mMaxZ;
}


float SPH::getInteractionRadius2() const
{
   return mHScaled2;
}


QList<Particle *>* SPH::getGrid()
{
   return mGrid;
}



vec3 SPH::getGravity() const
{
   return mGravity;
}


void SPH::setGravity(const vec3 &gravity)
{
   mGravity = gravity;
}


float SPH::getCflLimit() const
{
   return mCflLimit;
}


void SPH::setCflLimit(float cflLimit)
{
   mCflLimit = cflLimit;
   mCflLimit2 = mCflLimit * mCflLimit;
}


float SPH::getDamping() const
{
   return mDamping;
}


void SPH::setDamping(float damping)
{
   mDamping = damping;
}


float SPH::getTimeStep() const
{
   return mTimeStep;
}


void SPH::setTimeStep(float timeStep)
{
   mTimeStep = timeStep;
}


float SPH::getViscosityScalar() const
{
   return mViscosityScalar;
}


void SPH::setViscosityScalar(float viscosityScalar)
{
   mViscosityScalar = viscosityScalar;
}


float SPH::getStiffness() const
{
   return mStiffness;
}


void SPH::setStiffness(float stiffness)
{
   mStiffness = stiffness;
}
