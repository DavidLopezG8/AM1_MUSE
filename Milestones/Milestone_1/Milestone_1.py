from numpy import zeros, array, linspace
from numpy.linalg import norm
import matplotlib.pyplot as plt

#Método de Euler para integrar órbitas de kepler.
#Partimos de las condiciones iniciales
x=1
y=0
xdot=0
ydot=1

U_0= array([x, y, xdot, ydot])


#Definimos el numero de pasos y el delta t

Deltat=0.1
N=200

#Inicialimos los vectores que necesitamos para el método
F_euler=array( zeros( [N,4] ) )
U_euler=array( zeros( [N,4] ) )

U_euler[0,:]=U_0



#MEtodo de EULER: Un+1=Un+ deltat* Fn
for n in range(1 , N):
    
    F_euler[n-1,0] = U_euler[n-1,2]  
    F_euler[n-1,1] = U_euler[n-1,3]  
    F_euler[n-1,2] = -U_euler[n-1,0]/(norm(array([U_euler[n-1,0],U_euler[n-1,1]]))**3)
    F_euler[n-1,3] = -U_euler[n-1,1]/(norm(array([U_euler[n-1,0],U_euler[n-1,1]]))**3)

    U_euler[n,:] = U_euler[n-1,:] + Deltat * F_euler[n-1,:]



#Pasamos al metodo de Runge kutta

U_rk4=array( zeros( [N,4] ) )
K1_rk4=array( zeros( [N,4] ) )
K2_rk4=array( zeros( [N,4] ) )
K3_rk4=array( zeros( [N,4] ) )
K4_rk4=array( zeros( [N,4] ) )


U_rk4[0,:]=U_0

for n in range(1, N):

    # k1 = F(Un,tn)
    K1_rk4[n-1,0]= U_rk4[n-1,2]
    K1_rk4[n-1,1]= U_rk4[n-1,3]
    K1_rk4[n-1,2]= -U_rk4[n-1,0]/(norm(array([U_rk4[n-1,0],U_rk4[n-1,1]]))**3)
    K1_rk4[n-1,3]= -U_rk4[n-1,1]/(norm(array([U_rk4[n-1,0],U_rk4[n-1,1]]))**3)

    # k2 = F(Un + k1·(dt/2), tn + dt/2)
    K2_rk4[n-1,0] = U_rk4[n-1,2] + K1_rk4[n-1,2]*Deltat/2
    K2_rk4[n-1,1] = U_rk4[n-1,3] + K1_rk4[n-1,3]*Deltat/2
    K2_rk4[n-1,2] = - (U_rk4[n-1,0] + K1_rk4[n-1,0] * Deltat/2) / ((norm(array([U_rk4[n-1,0] + K1_rk4[n-1,0] * Deltat/2, U_rk4[n-1,1] + K1_rk4[n-1,3]*Deltat/2 ])))**3)
    K2_rk4[n-1,3] = - (U_rk4[n-1,1] + K1_rk4[n-1,1] * Deltat/2) / ((norm(array([U_rk4[n-1,0] + K1_rk4[n-1,0] * Deltat/2, U_rk4[n-1,1] + K1_rk4[n-1,3]*Deltat/2 ])))**3)
    
    # k3 = F(Un + k2·(dt/2), tn + dt/2)
    K3_rk4[n-1,0] = U_rk4[n-1,2] + K2_rk4[n-1,2]*Deltat/2
    K3_rk4[n-1,1] = U_rk4[n-1,3] + K2_rk4[n-1,3]*Deltat/2
    K3_rk4[n-1,2] = - (U_rk4[n-1,0] + K2_rk4[n-1,0] * Deltat/2) / ((norm(array([U_rk4[n-1,0] + K2_rk4[n-1,0] * Deltat/2, U_rk4[n-1,1] + K2_rk4[n-1,1]*Deltat/2 ])))**3)
    K3_rk4[n-1,3] = - (U_rk4[n-1,1] + K2_rk4[n-1,1] * Deltat/2) / ((norm(array([U_rk4[n-1,0] + K2_rk4[n-1,0] * Deltat/2, U_rk4[n-1,1] + K2_rk4[n-1,1]*Deltat/2 ])))**3)

    # k4 = F(Un + k3·dt, tn + dt)
    K4_rk4[n-1,0] = U_rk4[n-1,2] + K3_rk4[n-1,2]*Deltat
    K4_rk4[n-1,1] = U_rk4[n-1,3] + K3_rk4[n-1,3]*Deltat
    K4_rk4[n-1,2] = - (U_rk4[n-1,0] + K3_rk4[n-1,0]) / ((norm(array([U_rk4[n-1,0] + K3_rk4[n-1,0]*Deltat, U_rk4[n-1,1] + K3_rk4[n-1,1]*Deltat ])))**3)
    K4_rk4[n-1,3] = - (U_rk4[n-1,1] + K3_rk4[n-1,1]) / ((norm(array([U_rk4[n-1,0] + K3_rk4[n-1,0]*Deltat, U_rk4[n-1,1] + K3_rk4[n-1,1]*Deltat ])))**3)


    U_rk4[n,:]=U_rk4[n-1, :] + (Deltat/6) * (K1_rk4[n-1,:] + 2 * K2_rk4[n-1, :] + 2 * K3_rk4[n-1, :] + K4_rk4[n-1, :])


plt.plot( U_rk4[:,0] , U_rk4[:,1], "-")
plt.plot( U_euler[:,0] , U_euler[:,1], "-")
plt.show()