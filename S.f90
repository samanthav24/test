program stru
  implicit none

  real(8),allocatable:: x(:,:),y(:,:),kv(:,:),kn(:),uprov(:)
  integer:: ntot,nA,nB,Nstep,lbox,i,t,nnx,nny,nx,ny,c,ni,ik,Nk,r,num
  real(8):: qmin,pi,ki,a,b,f,j,w,f2,f3,f4,f5
  integer::g,h,ka
  integer :: k(50) = (/(i, i=1,500, 10)/)
  real(8), dimension(:), allocatable :: uk,SAA,SBB,SAB,cosA,sinA,cosB,sinB,SAAn,SBBn,SABn,S,Sn,cose,sino
  real(8)::min,max
  complex:: z,zs
  integer, dimension(:),allocatable::indk
  nA=325
  nB=175
  lbox= 20.412414523193153 !91.287092917527687  
  Nstep=1000
  ntot=nA+nB
  pi=3.1415926535
  qmin=(2*pi)/lbox
  num=50
  Nk=num**2
  allocate(x(1:Nstep,1:ntot),y(1:Nstep,1:ntot))
  allocate(kv(1:num**2,1:2),kn(1:num**2),uprov(1:num**2))
  allocate(SAA(1:num**2),SBB(1:num**2),SAB(1:num**2),cosA(1:num**2),cosB(1:num**2),sinA(1:num**2),sinB(1:num**2))
  allocate(S(1:num**2),cose(1:num**2),sino(1:num**2))
  open(unit=1,file="/Users/giuliajanzen/desktop/code/tr-aging.dat ",status="old")
  do t=1,Nstep
     write(*,*) t
     do i=1,ntot
        read(1,*) x(t,i),y(t,i)
     end do
  end do
  kv=0
  kn=0
  c=0

  do nnx=1,num
     nx=k(nnx)
     do nny=1,num
        ny=k(nny)
        kv(c,1)=nx*qmin
        kv(c,2)=ny*qmin
        kn(c)=sqrt(kv(c,1)**2+kv(c,2)**2)
        c=c+1
        
     end do
  end do
 

  min=minval(kn)-1
  max=maxval(kn)
  ni=0
  do while (min<max)
     ni = ni+1
     min = minval(kn, mask=kn>min)
     
     uprov(ni) = min
    
  enddo
  allocate(uk(1:ni))
  allocate(SAAn(1:ni),SBBn(1:ni),SABn(1:ni),Sn(1:ni))
 do i=1,ni
    
    uk(i)=uprov(i)

 end do

 do t=1,Nstep
 write(*,*) t  
    cosA=0
    cosB=0
    sinA=0
    sinB=0
    cose=0
    sino=0
    do ik=1,Nk

       do i=1,ntot
          cose(ik)=cose(ik)+cos((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
          sino(ik)=sino(ik)+sin((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
          if (i<nA) then
             cosA(ik)=cosA(ik)+cos((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
             sinA(ik)=sinA(ik)+sin((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
          else if(i>=nA) then
             cosB(ik)=cosB(ik)+cos((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
             sinB(ik)=sinB(ik)+sin((kv(ik,1)*x(t,i)+kv(ik,2)*y(t,i)))
          end if
       end do
       SAA(ik)=SAA(ik)+((cosA(ik))**2+(sinA(ik))**2)/nA
       SBB(ik)=SBB(ik)+((cosB(ik))**2+(sinB(ik))**2)/nB
       a=cosB(ik)
       b=sinB(ik)
       z=complex(a,b)
       zs=conjg(z)
       SAB(ik)=SAB(ik)+real(complex(cosA(ik),sinA(ik))*zs)/(sqrt(real(nA)*real(nB)))
       S(ik)=S(ik)+((cose(ik))**2+(sino(ik))**2)/ntot
    end do
    
 end do
 SAA=SAA/Nstep
 SBB=SBB/Nstep
 SAB=SAB/Nstep
 S=S/Nstep

  


 do i=1,ni
    r=0
    f=0
    j=0
    w=0
    a=0
    do g=1,Nk
      
       if(uk(i)==kn(g)) then
          r=r+1
          f=f+SAA(g)
          j=j+SBB(g)
          w=w+SAB(g)
          a=a+S(g)
       end if
    end do
    SAAn(i)=f/real(r)
    SBBn(i)=j/real(r)
    SABn(i)=w/real(r)
    Sn(i)=a/real(r)
 end do
 f=real(nA)/real(ntot)
 f2=real(nB)/real(ntot)
 f3=sqrt(real(nB)*real(nA))/real(ntot)
 !SAAn=SAAn*f
! SBBn=SBBn*f2
 !SABn=SABn*f3
 !f4=real(nA)/real(ntot)
 !f5=real(nB)/real(ntot)
  open(unit=7,file="/Users/giuliajanzen/desktop/SA-ex0.dat",status="replace")
 do ik=2,ni
    write(7,*) uk(ik),SAAn(ik)
 end do
 close(unit=7)
  open(unit=8,file="/Users/giuliajanzen/desktop/SB-ex0.dat",status="replace")
 do ik=2,ni
    write(8,*) uk(ik),SBBn(ik)
 end do
 close(unit=8)
  open(unit=9,file="/Users/giuliajanzen/desktop/SAB-ex0.dat",status="replace")
 do ik=2,ni
    write(9,*) uk(ik),SABn(ik)
 end do
 close(unit=9)
 open(unit=10,file="/Users/giuliajanzen/desktop/St-ex0a.dat",status="replace")
 do ik=2,ni
    write(10,*) uk(ik),Sn(ik)
 end do
 close(unit=10)
  
end program stru
