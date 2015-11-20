/*
  particle_system.cl
  Austin Brennan
  10/8/2015

  Basic particle system kernel. Gravity pulls down particles
  and the particles bounce off of two spheres. When the particle
  bounces off of a sphere it's color changes to either orange or
  white depending on which sphere it hit.
*/


#define MULT (87.0f)
#define MOD (3647.0f)

float goober(float prev)
{
  prev *= (MOD*MULT);
  return(fmod(prev,MOD)/MOD);
}



#define STEPS_PER_RENDER 30
#define MASS 1.0f
#define DELTA_T (0.002f)
#define FRICTION 0.4f
#define RESTITUTION 0.1f

#define EPS_DOWN (-0.4f) // (-0.2f) // gravity
#define V_DRAG (4.0f)


// basic gravitational force
float4 getforce(float4 pos, float4 vel)
{
  float4 force;
  force.x = 0.0f;
  force.y = EPS_DOWN-V_DRAG*vel.y;
  force.z = 0.0;
  force.w = 1.0f;
  return(force);
}

float4 calculateVout(float4 vin, float radius, float4 compDiff)
{
  float mylength;
  float4 normal;

  mylength = sqrt(vin.x*vin.x + vin.y*vin.y + vin.z*vin.z);
  vin.x = vin.x/mylength;
  vin.y = vin.y/mylength;
  vin.z = vin.z/mylength;
  vin.w = 0.0f;
  normal = compDiff/radius; // unit normal vector
  normal.w = 0.0f;

  return (vin-(1.0f+RESTITUTION)*(dot(vin, normal)*normal)) * mylength;
}

__kernel void VVerlet(__global float4* p, __global float4* v, __global float* r, __global float4* color)
{
  unsigned int i = get_global_id(0);
  float4 force, zoom, compDiff; // component difference between ball and p
  float4 positionOfBall1 = (float4)(-0.2f, 0.4f, -0.2f, 0.3f); // (x, y, z, radius)
  float4 positionOfBall2 = (float4)(0.1f, 0.2f, 0.2f, 0.2f);
  float mylength, distanceToBall;
  for(int steps=0;steps<STEPS_PER_RENDER;steps++){
    force = getforce(p[i],v[i]);
    v[i] += force*DELTA_T/2.0f;
    p[i] += v[i]*DELTA_T;
    force = getforce(p[i],v[i]);
    v[i] += force*DELTA_T/2.0f;
    if(p[i].y<0.0f) {
      // regenerate position, velocity, and color
      zoom.x = r[i]-0.5f;
      r[i] = goober(r[i]);
      zoom.y = 0.2f*r[i]+0.8f;
      r[i] = goober(r[i]);
      zoom.z = r[i]-0.5f;
      p[i] = zoom;
      r[i] = goober(r[i]);
      zoom.x = 0.05f*r[i];
      r[i] = goober(r[i]);
      zoom.y = 0.05f*r[i];
      r[i] = goober(r[i]);
      zoom.z = 0.05f*r[i];
      v[i] = zoom;
      r[i] = goober(r[i]);
      color[i]=(float4)(0.917f,0.415f,0.125f,1.0f);
      }
    else{
      // Check for ball collision.
      compDiff =  p[i] - positionOfBall1;
      distanceToBall = sqrt(compDiff.x*compDiff.x+compDiff.y*compDiff.y+compDiff.z*compDiff.z);
      if(distanceToBall < positionOfBall1.w) {
        p[i].y += 0.01f;
        v[i] = calculateVout(v[i], positionOfBall1.w, compDiff);
        v[i].w = 1.0;

        color[i] = (float4)(0.642f,0.352f,1.0f,1.0f);
       }
      compDiff =  p[i] - positionOfBall2;
      distanceToBall = sqrt(compDiff.x*compDiff.x+compDiff.y*compDiff.y+compDiff.z*compDiff.z);
      if(distanceToBall < positionOfBall2.w) {
        p[i].y += 0.01f;
        v[i] = calculateVout(v[i], positionOfBall2.w, compDiff);
        v[i].w = 1.0;

        color[i] = (float4)(1.0f,1.0f,1.0f,1.0f);
      }
    }
  }
  p[i].w = 1.0f;
}
