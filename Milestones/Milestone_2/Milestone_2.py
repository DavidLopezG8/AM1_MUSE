from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Problema de Kepler
def Kepler(U, t): # U = vector de 4 componentes

    r = U[0:2]
    rdot = U[2:4]
    F = concatenate( (rdot,-r/norm(r)**3), axis=0 )

    return F

#ESQUEMAS NUMÉRICOS

#  EULER explícito
def Euler(F, U, dt, t):

    return U + dt * F(U,t)

# Runge-Kutta 4
def RK4(F, U, dt, t):

    k1 = F(U,t)
    k2 = F( U + k1 * dt/2, t + dt/2)
    k3 = F( U + k2 * dt/2, t + dt/2)
    k4 = F( U + k3 * dt  , t + dt  )
    
    return U + dt/6 * ( k1 + 2*k2 + 2*k3 + k4)

#  Cauchy
#
# Obtener la solución de un problema dU/dt = F (ccii), dada una CI
# y un esquema temporal
#
# Inputs : 
#          -> Esquema temporal
#          -> Funcion F(U,t)
#          -> Condición inicial
#          -> Partición temporal
#
# Output :  
#          -> Solución para todo "t" y toda componente
#
def Cauchy(Esquema, F, U0, t):
    
    N = len(t)-1
    U = zeros((N+1, len(U0)))

    U[0,:] = U0
    for n in range(0,N):
        U[n+1,:] = Esquema( F, U[n,:], t[n+1]-t[n], t[n] )

    return U 



###########################################################
#                          DATOS                          #
###########################################################
# Selecciona el problema que quieres resolver (de los implementados en las funciones)
problema = Kepler

# Condiciones iniciales
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1


# Instantes inicial y final
t0 = 0
tf = 20

# Número de intervalos (=nº de instantes de tiempo - 1)
N = 200




#Defino el delta de t
t = zeros(N+1)
t = linspace(t0,tf,N+1)
dt = (tf-t0)/N



# Creamos vector de condiciones iniciales
U0 = array( [x0,y0,vx0,vy0] )

# Soluciones

U_euler = Cauchy(Euler, kepler, U0, t)
U_rk4   = Cauchy(RK4  , Kepler, U0, t)




plt.plot( U_euler[:, 0], U_euler[:,1] , '-b' , lw = 1.0, label ="Euler explícito" )
plt.plot( U_rk4[:, 0]  , U_rk4[:,1]   , '-r' , lw = 1.0, label ="Runge-Kutta 4" )
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



