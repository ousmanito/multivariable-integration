---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Intégrales avancées

## Exercice 1 : intégration de Gauss-Legendre

On se propose d'évaluer numériquement l'intégrale suivante par deux méthodes :

$$I = \int\limits_{-\pi}^{\pi} \cos^2(x) {\rm d}x $$

- Utiliser la fonction quad du module `integrate` de `scipy` et vérifier qu'on obtient bien la valeur attendue $\pi$.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 36591f1950a8d590a5ef535a1a03b3f1
  grade: false
  grade_id: cell-65f324a4720645fd
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "034b128a1e405704475c1d75ca6df64f", "grade": false, "grade_id": "cell-7014596865b03f3d", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": true}}

- Coder votre algorithme d'intégration par la méthode de Gauss-Legendre pour $N=10$ pas d'échantillonage. L'appliquer à l'intégrale ci-dessus.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 37a0d7f3e7eb94d247efce3cae68072e
  grade: false
  grade_id: cell-2515b32950fdc268
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# Determination des poids et des noeuds

N = 10
x,w = np.polynomial.legendre.leggauss(N)

#Integration par la méthode de Gauss Legendre

J = np.sum(np.pi*f(np.pi*x)*w)
print(J)

#On retrouve bien pi
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fa23a0600e9144e9af75dcd2d9873685", "grade": false, "grade_id": "cell-6bba2e43c1d526c7", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": true}}

- Evaluer et représenter dans un graphique l'erreur relative commise en fonction de $N \in [2, 100]$. Commenter le résultat.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 6b86e8f1812cd596fdc3747ffe84bf22
  grade: false
  grade_id: cell-cfab152eec366946
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "204726dc129757c624a6c2cd30999efc", "grade": false, "grade_id": "cell-5c211aa781632702", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

## Exercice 2 : calcul d'une intégrale à 2 dimensions

On se propose d'évaluer numériquement l'intégrale suivante par différentes méthodes :

$$I = \int\limits_{x=\pi}^{2\pi} \int\limits_{y=0}^{\pi} \left[y \sin(x) + x \cos(y)\right]{\rm d}x {\rm d}y $$

- Représenter la fonction pour $x\in [-10\pi, 10\pi], y\in [-10\pi, 10\pi]$.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 681a7239bfb948a0cf7bb220fd031f56
  grade: false
  grade_id: cell-129e2b5032877940
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6e66b8200d09b890a4f0a3150b681f9b", "grade": false, "grade_id": "cell-3268300f667b1614", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

- Utiliser la fonction dblquad du module `integrate` de `scipy` pour obtenir une valeur de référence.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: e5ecee8cf02a97c3a2d7cf2a02224b71
  grade: false
  grade_id: cell-652523070f08ee62
  locked: false
  schema_version: 3
  solution: true
  task: false
---
def f1(y,x):
    return y*np.sin(x)+x*np.cos(y)

I = integrate.dblquad(f1,np.pi,2*np.pi,0,np.pi)
I
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "56327a8d4cb59a80d1fb3e3e8719b3f6", "grade": false, "grade_id": "cell-909ef68d403cd177", "locked": true, "points": 9, "schema_version": 3, "solution": false, "task": true}}

- Evaluer cette intégrale en écrivant vos propres algorithmes pour la méthode des rectangles, de Gauss-Legendre et Monte-Carlo.

    - Méthode des rectangles :
$$I= \frac{V}{N_x N_y}\sum_{i=1}^{N_x} \sum_{j=1}^{N_y} f(x[i],y[j]) $$

    - Méthodes de Gauss-Legendre :
$$I= \frac{V}{4}\sum_{i=1}^N \sum_{j=1}^N f(x_p[i],y_p[j])w_x[i]w_y[j]$$

    - Méthodes Monte-Carlo :
$$I= \frac{V}{N}\sum_{i=1}^N f(x[i],y[i])$$

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 4910f976a4a1e966f55060d74f708264
  grade: false
  grade_id: cell-56791223f3949227
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c2dfc4204defa51e96c6b8ae0f2645d4", "grade": false, "grade_id": "cell-a82c704c3176f6f5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Commenter la précision et le temps approximatif d'exécution de ces différentes méthodes.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "69c2f142fa6da40de5193ca17eb8b7a2", "grade": false, "grade_id": "cell-6c94c0308780e461", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Exercice 3 : tracé et calcul de l'aire d'une ellipse par méthode Monte-Carlo

Une ellipse est définie par la taille de ces deux axes suivant des axes orthogonaux $x$ (demi-axe $a$) et $y$ (demi-axe $b$). L'ellipse vérifie alors l'équation:

$$ \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 =1$$ 

On peut alors tracer l'ellipse de manière paramétrique de la façon suivante :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b53f8fa6f988615bc870b29ac2241a2d
  grade: false
  grade_id: cell-47bd98633eebd338
  locked: true
  schema_version: 3
  solution: false
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "06f63431bba8834c8d7ab5c4693c74a7", "grade": false, "grade_id": "cell-97b55399d953b3f6", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

- Tirer, de manière aléatoire et avec une distribution uniforme, des points dans le plan dans une gamme au moins aussi large le rectangle $[-a,a]\times [-b,b]$. 
- Afficher ces points comme marqueurs par dessus l'ellipse.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 5b755711ea2250e2a629826f79d7bae6
  grade: false
  grade_id: cell-a1fbf34a5095e061
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f83af9412b5aefab0f2dc39ac485ae95", "grade": false, "grade_id": "cell-33f3dfbf2d301454", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

- Établisser un critère mathématique pour déterminer si un point est à l'intérieur de l'ellipse et indiquer respectivement par des marqueurs verts et rouges les points à l'intérieur et à l'extérieur de l'ellipse. On pourra utiliser `np.where`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: d8affd4f388429bf95595a0254f1fd48
  grade: false
  grade_id: cell-3ee1f982fe3189db
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b49f1c2a4831316e1e1612613167b7e6", "grade": false, "grade_id": "cell-44b2a2266afd58b3", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

- Calculer l'aire de l'ellipse par intégration Monte-Carlo en utilisant $10^4$ points et comparez à la valeur analytique attendue.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 3347cf64a8a1239e66ccd6a8588054a5
  grade: false
  grade_id: cell-20739f01afdcdcba
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "83ebc80d43df4620ddeb7c92c3646ae0", "grade": false, "grade_id": "cell-2cc886f51f12f286", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": true}}

## Exercice 4 : calcul du volume d'hypersphères par la méthode Monte-Carlo
Dans un espace à d dimensions, une sphère de rayon $R$ est définie par (voir par exemple la page wikipedia [n-sphères](https://fr.wikipedia.org/wiki/N-sph%C3%A8re)):

$$ \sum_{i=1}^{d} x_{i}^2=R^2$$ 

- Ecrire une fonction utilisant la méthode Monte-Carlo pour évaluer le volume d'une hypersphère à d dimensions.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 364e35a23e3dcbc3248ff3b9d0697535
  grade: false
  grade_id: cell-42e1a0327a262e4d
  locked: false
  schema_version: 3
  solution: true
  task: false
---
N = 10**4
def vol_hyp(R,N,d):
    xi = np.random.uniform(-R,R,size=(N,d))
    norme_carré = [np.sum(xi[i]**2) for i in range(np.shape(xi)[0])]
    in_hyp=[i for i in norme_carré if i < R**2]
    return len(in_hyp)*((2*R)**d)/N
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e552b9849c7b2ad24770541e4c5460fd", "grade": false, "grade_id": "cell-357132859e5660c0", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": true}}

- Calculer ce volume pour une hypersphère de rayon = 1 et en dimension n=1, 2, 3 et 4.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: bc1d7ca8d1ecd7208c70a960c04e868a
  grade: false
  grade_id: cell-de95006c98b175cb
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# Volumes en dimension 1,..,4
v_mc = [vol_hyp(1,N,d) for d in range(1,5)]
v_mc
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "471180f7f8922797838a5e2d2f45c8cc", "grade": false, "grade_id": "cell-9f87040a70923e72", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

- Comparer les valeurs obtenues aux valeurs exactes, vous pourrez utiliser la fonction gamma du module scipy.special. 

$$ V=\frac{\pi^{n/2}}{\Gamma(n/2+1)}$$

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 7c4b883de7d1eae4551d935f47843331
  grade: false
  grade_id: cell-93a659f510135851
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

```{code-cell} ipython3

```
