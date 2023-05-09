
# Intégrales avancées

## Exercice 1 : intégration de Gauss-Legendre

On se propose d'évaluer numériquement l'intégrale suivante par deux méthodes :

$$I = \int\limits_{-\pi}^{\pi} \cos^2(x) {\rm d}x $$

- Utiliser la fonction quad du module `integrate` de `scipy` et vérifier qu'on obtient bien la valeur attendue $\pi$.

```python

#Importation des  modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import style

style.use('dark_background')

def f(x):
    return np.cos(x)**2

#Integration de f grace à quad
I = integrate.quad(f,-np.pi,np.pi)

print(I)
```

- Coder votre algorithme d'intégration par la méthode de Gauss-Legendre pour $N=10$ pas d'échantillonage. L'appliquer à l'intégrale ci-dessus.

```python
# Determination des poids et des noeuds

N = 10
x,w = np.polynomial.legendre.leggauss(N)

#Integration par la méthode de Gauss Legendre

J = np.sum(np.pi*f(np.pi*x)*w)
print(J)

#On retrouve bien pi
```

- Evaluer et représenter dans un graphique l'erreur relative commise en fonction de $N \in [2, 100]$. Commenter le résultat.

```python

#Erreur relative et abcisse

y = [(np.abs(J-I[0]))/np.pi for N in range(2,101)]
x = np.arange(2,101)

# Graphique

plt.plot(x,y)
plt.xlabel('Nombre de points')
plt.ylabel('Erreur relative')
plt.title('Erreur en fonction de N')
plt.show()

#L'erreur est constante et toujours égale à  3,5 * 10⁻9 ou 3,5*10⁻7 %, on en déduit que la quadrature de
# Gauss Legendre est très précise.
```



## Exercice 2 : calcul d'une intégrale à 2 dimensions

On se propose d'évaluer numériquement l'intégrale suivante par différentes méthodes :

$$I = \int\limits_{x=\pi}^{2\pi} \int\limits_{y=0}^{\pi} \left[y \sin(x) + x \cos(y)\right]{\rm d}x {\rm d}y $$

- Représenter la fonction pour $x\in [-10\pi, 10\pi], y\in [-10\pi, 10\pi]$.

```python
#Fonction 
def f(x,y):
    return y*np.sin(x)+x*np.cos(y)


#Abcisse et ordonnée
x = np.linspace(-10*np.pi,10*np.pi,100)
y = x

#Représentation graphique
xx,yy = np.meshgrid(x,y)
z = f(xx,yy)

fig = plt.figure()
im = plt.imshow(z,origin='lower',extent=[np.pi,2*np.pi,0,np.pi])
c = fig.colorbar(im)
plt.xlabel("x")
plt.ylabel('y')
c.set_label("z")
plt.title("Tracé de la fonction")
plt.show()
```


- Utiliser la fonction dblquad du module `integrate` de `scipy` pour obtenir une valeur de référence.

```python

def f1(y,x):
    return y*np.sin(x)+x*np.cos(y)

I = integrate.dblquad(f1,np.pi,2*np.pi,0,np.pi)
I
```


- Evaluer cette intégrale en écrivant vos propres algorithmes pour la méthode des rectangles, de Gauss-Legendre et Monte-Carlo.

    - Méthode des rectangles :
$$I= \frac{V}{N_x N_y}\sum_{i=1}^{N_x} \sum_{j=1}^{N_y} f(x[i],y[j]) $$

    - Méthodes de Gauss-Legendre :
$$I= \frac{V}{4}\sum_{i=1}^N \sum_{j=1}^N f(x_p[i],y_p[j])w_x[i]w_y[j]$$

    - Méthodes Monte-Carlo :
$$I= \frac{V}{N}\sum_{i=1}^N f(x[i],y[i])$$

```python3
import time

#Methode des rectangles

def rect(f,N):
    
    hx = np.pi/N
    hy = np.pi/N
    
    xi = [ np.pi+k*hx for k in range(N) ]
    yi = [ k*hy for k in range(N) ]
    
    xx,yy = np.meshgrid(xi,yi)
    z = f(xx,yy)
    
    return np.sum(z*(np.pi**2)/N**2)

depart_rect = time.time()

print(rect(f,100))

fin_rect = time.time()

print('temps :',fin_rect-depart_rect,'s')

#Methode de Gauss Legendre

def f2(x,y,wx,wy):
    return f(x,y)*wx*wy

def gaussl(f,N):
    x, wx = np.polynomial.legendre.leggauss(N)
    y, wy = np.polynomial.legendre.leggauss(N)
    
    xx,yy = np.meshgrid(x,y)
    z = f(xx*(np.pi/2)+(3*np.pi/2),yy*(np.pi/2)+(np.pi/2))
    return (np.pi**2/4)*np.sum(z*wx*wy)

depart_gl = time.time()
print(gaussl(f,2))
fin_gl = time.time()
print('temps :',fin_gl-depart_gl,'s') 

#Methode Monte Carlo

def mc(f,N):

    x = np.random.uniform(np.pi,2*np.pi,size=N)
    y = np.random.uniform(0,np.pi,size=N)
    z = f(x,y)
    return np.sum(z)*(np.pi**2)/N

depart_mc = time.time()
print(mc(f,100))
fin_mc = time.time()
print('temps :',fin_mc-depart_mc,'s')
```

- Commenter la précision et le temps approximatif d'exécution de ces différentes méthodes.



## Exercice 3 : tracé et calcul de l'aire d'une ellipse par méthode Monte-Carlo

Une ellipse est définie par la taille de ces deux axes suivant des axes orthogonaux $x$ (demi-axe $a$) et $y$ (demi-axe $b$). L'ellipse vérifie alors l'équation:

$$ \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 =1$$ 

On peut alors tracer l'ellipse de manière paramétrique de la façon suivante :

```python
import matplotlib.pyplot as plt
import numpy as np


def ellipse(a,b,n):
    t=np.linspace(0,2*np.pi,n)
    x=a*np.cos(t)
    y=b*np.sin(t)
    return x,y

n = 1000
a, b = 1, 3
xell, yell = ellipse(a,b,n)

fig = plt.figure()
plt.plot(xell, yell)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```


- Tirer, de manière aléatoire et avec une distribution uniforme, des points dans le plan dans une gamme au moins aussi large le rectangle $[-a,a]\times [-b,b]$. 
- Afficher ces points comme marqueurs par dessus l'ellipse.

```python
x_r = np.random.uniform(-a,a,size=N)
y_r = np.random.uniform(-b,b,size=N)
N=10**4

fig1 = plt.figure()
plt.plot(xell, yell)
plt.scatter(x_r,y_r,color='b')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

- Établisser un critère mathématique pour déterminer si un point est à l'intérieur de l'ellipse et indiquer respectivement par des marqueurs verts et rouges les points à l'intérieur et à l'extérieur de l'ellipse. On pourra utiliser `np.where`.

```python

#Utilisation de np.where pour trouver les points contenus dans l'ellipse.
in_ell = np.where(((x_r**2)/a**2)+((y_r**2)/b**2)<1)
out_ell = np.where(((x_r**2)/a**2)+((y_r**2)/b**2)>=1)

#Graphique
fig1 = plt.figure()
plt.plot(xell, yell)
plt.scatter(x_r[in_ell],y_r[in_ell],color='g')
plt.scatter(x_r[out_ell],y_r[out_ell],color='r')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

- Calculer l'aire de l'ellipse par intégration Monte-Carlo en utilisant $10^4$ points et comparez à la valeur analytique attendue.

```ipython

#Aire de l'ellipse gràace à la méthode MC
aire = (12)*(np.shape(in_ell)[1])/N
print(aire)

#Aire d'une ellipse 
aire_theo = np.pi*a*b
print(aire_theo)

#Erreur relative 

err = np.abs(aire_theo-aire)/aire_theo
print(err*100,'%')

#C'est très précis
```


## Exercice 4 : calcul du volume d'hypersphères par la méthode Monte-Carlo
Dans un espace à d dimensions, une sphère de rayon $R$ est définie par (voir par exemple la page wikipedia [n-sphères](https://fr.wikipedia.org/wiki/N-sph%C3%A8re)):

$$ \sum_{i=1}^{d} x_{i}^2=R^2$$ 

- Ecrire une fonction utilisant la méthode Monte-Carlo pour évaluer le volume d'une hypersphère à d dimensions.

```python

N = 10**4
def vol_hyp(R,N,d):
    xi = np.random.uniform(-R,R,size=(N,d))
    norme_carré = [np.sum(xi[i]**2) for i in range(np.shape(xi)[0])]
    in_hyp=[i for i in norme_carré if i < R**2]
    return len(in_hyp)*((2*R)**d)/N
```

- Calculer ce volume pour une hypersphère de rayon = 1 et en dimension n=1, 2, 3 et 4.

```python
# Volumes en dimension 1,..,4
v_mc = [vol_hyp(1,N,d) for d in range(1,5)]
v_mc
```

- Comparer les valeurs obtenues aux valeurs exactes, vous pourrez utiliser la fonction gamma du module scipy.special. 

$$ V=\frac{\pi^{n/2}}{\Gamma(n/2+1)}$$

```python

# Utlisation de la fonction gamma afin de calculer les volumes
from scipy.special import gamma
def vol_gam(R,d):
    return ((np.pi**(d/2))*(R**d))/(gamma((d/2)+1))

# Volumes en dimension 1,..,4
v_g = [vol_gam(1,d) for d in range(1,5)]
print(v_g)

#Erreur relative pour chaque dimension (%)

err = np.abs(np.array(v_g)-np.array(v_mc))/np.array(v_g)*100
print(err)
    
```
