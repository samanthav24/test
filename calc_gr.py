#!/usr/bin/env python
# Calculate radial distribution function from LAMMPS output
#
# See also:
# Allen & Tildesley, Computer Simulation of Liquids, p. 183-184.
#
# LMCJ, 02-02-2015


#import numpy as np
#from StringIO import StringIO
from numpy import *
import matplotlib.pyplot as plt

# LOAD DATA

#f = file("kalj_T0.45_n1900_v1583.333_e25600000_1_s25600000_1.atom","r")
#f = file("testfile1.atom","r")
box      = loadtxt("box.atom")
N        = int(loadtxt("num.atom"))
time     = [int(x) for x in loadtxt("timestep.atom")]
coord    = loadtxt("coord.atom")

#box       = zeros((3,2))                     # matrix with pre-known dimensions
#time      = []                               # empty list that will grow during loop
#coord     = []                               # empty array that will grow during loop
#N         = 0
#nstep     = 0
#jj        = 0
#for i in f:
#  # extract time steps
#  if jj==(N+9)*nstep+1:                      # first iteration: jj==1
#    time.append(int(i)) 
#    nstep = nstep+1
#  # extract #atoms and box size
#  elif jj==3:
#    N = int(i[0:-1])                         # number of atoms
#  elif jj>=5 and jj<=7:
#    c = [float(x) for x in i.split(' ')]
#    box[jj-5,0:2] = c                        # box dimensions
#    #c = StringIO(i)
#    #box[jj-5,0:2] = loadtxt(c)
#  # extract particle coordinates
#  elif jj>(nstep-1)*(N+9)+8 and jj<nstep*(N+9):
#    c = [float(x) for x in i.split(' ')]
#    coord = append(coord,c,axis=0)           # this will create a long 1-D list
#  jj = jj+1
#coord = coord.reshape(nstep*N,5)             # reshape into 2-D matrix array
#f.close()

if len(coord)/N != len(time):
  print('Size of coordinates-array inconsistent with (#atoms)*(#timesteps).')

Lx        = box[0,1]-box[0,0];
Ly        = box[1,1]-box[1,0];
Lz        = box[2,1]-box[2,0];

iA        = nonzero(coord[0:N,1] - 1)[0]     # particles of type A
iB        = nonzero(coord[0:N,1] - 2)[0]     # particles of type B
NA        = len(iA)
NB        = len(iB)
#iA        = list(iA)                         # convert tuple to list
#iB        = list(iB)                         # convert tuple to list
#siA       = shape(iA)
#siB       = shape(iB)
#NA        = siA[1]
#NB        = siB[1]

if NA+NB != N:
  print('Size of NA+NB inconsistent with N.')


# DEFINE R-GRID

rmax      = 5
dr        = 0.02
r         = arange(0,rmax+dr,dr)             # 'arange' works for floats ('range' for integers)
NR        = len(r)
grAA      = zeros((NR,1))
grBB      = zeros((NR,1))
grAB      = zeros((NR,1))


# CALCULATE G(R)

trange    = range(1,100,5)                   # range(1,b,c): timestep #1 to (b-1) in steps of c

for it in range(len(trange)):                # loop over time steps

  iit     = trange[it]
  coordt  = coord[(iit-1)*N:iit*N,:]

  # loop over all particle pairs
  for i in range(N):                         # i = particle 1 to N-1
    for j in range(i+1,N):                   # j = particle i+1 to N (so that i~=j)

      rijx = abs(coordt[i,2]-coordt[j,2])
      rijy = abs(coordt[i,3]-coordt[j,3])
      rijz = abs(coordt[i,4]-coordt[j,4])

      # periodic boundary conditions --> rij = min( rij, abs(rij-L) )
      if rijx > 0.5*Lx:
        rijx = abs(rijx-Lx)
      if rijy > 0.5*Ly:
        rijy = abs(rijy-Ly)
      if rijz > 0.5*Lz:
        rijz = abs(rijz-Lz)

      rij  = sqrt(rijx**2 + rijy**2 + rijz**2)

      if rij <= rmax:
        igr = round(rij/dr)                  # igr can range from 0 to NR-1

        if i in iA and j in iA:
          grAA[igr] = grAA[igr] + 2          # factor of 2 from i--j + j--i
        elif i in iB and j in iB:
          grBB[igr] = grBB[igr] + 2
        elif (i in iA and j in iB) or (i in iB and j in iA):
          grAB[igr] = grAB[igr] + 1


# NORMALIZE G(R)

V         = Lx*Ly*Lz
const     = 4*pi/3
dr3       = zeros((NR,1))
for ir in range(NR):
  rlow    = ir*dr
  rup     = rlow + dr
  dr3[ir] = rup**3-rlow**3

nidealA   = const*(NA*NA/V)*dr3              # g(r) for ideal gas of A particles
nidealB   = const*(NB*NB/V)*dr3              # g(r) for ideal gas of B particles
nidealAB  = const*(NA*NB/V)*dr3              # g(r) for ideal gas of A+B particles

grAA_norm = (grAA/nidealA)/len(trange) 
grBB_norm = (grBB/nidealB)/len(trange)
grAB_norm = (grAB/nidealAB)/len(trange)

gr_all    = hstack([r.reshape(NR,1), grAA_norm, grBB_norm, grAB_norm])


# WRITE TO FILE
g         = open('gr.dat','w')
#print >>g box[1,1]
savetxt(g, gr_all, fmt='%.12e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
g.close()


# PLOT RESULTS
g        = loadtxt('gr.dat')
plt.plot(g[:,0],g[:,1],'b-s',g[:,0],g[:,2],'r-s',g[:,0],g[:,3],'g-^')
plt.show()


