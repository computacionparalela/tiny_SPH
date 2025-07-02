// base
#include "particle.h"

Particle::Particle()
 : mMass(1.0f),
   mDensity(0.0f)
{
   mVelocity.set(0.0f, 0.0f, 0.0f);
   mAcceleration.set(0.0f, 0.0f, 0.0f);
}


