#Importa biblioteca numpy para armazenar os vetores
import numpy as np
#Carrega os vetore com eixo x e y
x = np.array([1.0,2.0,3.0,4.0,5.0])
y = np.array([1.0,2.1,2.8,4.1,5.2])

#Importa biblioteca matplotlib para plotar no grafico os dados
from matplotlib import pyplot as pl
pl.plot(x,y,'o')

#importa biblioteca scipy para usar a regressao linear
from scipy.stats import linregress
def lin_regression(x,y):
    m, b, R, p, SEm = linregress(x,y)

    n = len(x)
    SSx = np.var(x,ddof=1)*(n-1)
    SEb2 = SEm**2 * (SSx/n + np.mean(x)**2)
    SEb = SEb2**0.5

    return m, b, SEm, SEb, R, p

m, b, Sm, Sb, R, p = lin_regression(x, y)

print('m = {:>.4g} +- {:6.4f}'.format(m, Sm))
print('b = {:>.4g} +- {:6.4f}\n'.format(b, Sb))

print('R2 = {:7.5f}'.format(R**2))
print('p of test F : {:<8.6f}'.format(p))

pl.plot(x,y, 'o')
pl.xlim(0,None)
pl.ylim(0, None)

# desenho da recta, dados 2 pontos extremos
# escolhemos a origem e o max(x)
x2 = np.array([0, max(x)])

pl.plot(x2, m * x2 + b, '-')

# Anotação sobre o gráfico:
ptxt = 'm = {:>.4g} +- {:6.4f}\nb = {:>.4g} +- {:6.4f}\nR2 = {:7.5f}'

t = pl.text(0.5, 4, ptxt.format(m, Sm, b, Sb, R**2), fontsize=14)
pl.show()
