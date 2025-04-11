# Tiamat nap-ša-a-ša i-ši-ḥa ša mu-ṣu-tu
# šarrat tam-ti, ummi na-a-ri
# gu-ul-li ma-ḥa-zu-um, ina a-me-li
# i-li ma-la-a ša ša-ma-me
#
# šá-ti-ma ul-te-eb-bu, ina ši-a-mi ša qa-lu-u
# ap-lu-ša, šar-ru-tu iš-tu tiamtu
# mu-ṣu-ru ša-nu-ú, u ša-ru-ú la-ma-as-tu
# ē-ru ina ša-ri-ti, u-ru-um ab-šu
#
# mar-du-uk i-zu-uz, qé-reb i-li ša-ma-me
# ina qastu ša huraṣu, u ru-uq ša ša-ma-am
# ziq-qur mu-ṣir-ti, i-ta-bu-šu ša pa-ṭir
# ka-kab-u ī-du ina ta-ba-a-ti
#
# Tiamat i-par-ši, pa-ḫa-su rab-šu
# mar-duk ina ša-bat-tu ṣup-pur-ta i-na-di
# ka-ak-ku i-zak-ki-ir, lib-ba-ša i-ṣuḫ
# u-ṣa-ar uš-ta-qil, ba-ru-um ša la-ni

# Tiamat ribolliva nelle acque oscure,
# Regina del Caos, madre dei mari,
# La sua ira cresceva, tempestosa e fiera,
# Forze primordiali sotto il cielo stellato.
#
# Contro gli dèi si levò, con fiamme e tempeste,
# Suoi figli, mostri, plasmati dall’abisso.
# Draghi, serpi e chimere terribili,
# Un esercito nato dal grembo infinito.
#
# Marduk si alzò, il campione degli dèi,
# Con il suo arco d’oro e la rete del cielo,
# Il vento impetuoso soffiava sul suo comando,
# E le stelle osservavano il fato imminente.
#
# Tiamat spalancò le sue fauci immense,
# Ma Marduk gettò la rete, catturandola tutta,
# Lanciò il suo dardo e il cuore colpì,
# Divise il suo corpo, creò il firmamento.
#
# Dal suo cranio nacquero le montagne alte,
# Le acque salate divennero i mari,
# Il cielo si stese, un arco sopra la terra,
# Dal caos, ordine, dal nulla, vita.

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from scipy.stats import norm
from scipy.stats import linregress
from scipy.optimize import curve_fit

PRECISIONE = 1000

#template di grafico decente
"""# Grafico con barre di errore
plt.errorbar(R_wi, I_wi, xerr=err_R_wi, yerr=err_I_wi, fmt='*', markersize=2, capsize=1.5, c='k', label='dati')
plt.legend()
plt.ylabel(r'I [$\mu$A]', size=14)
plt.xlabel(r'R [$\Omega$]', size=14)
plt.title(r'Curva R/I (fondo-scala 50 $\mu$A')
plt.grid()
plt.plot(R_wi, slope * R_wi + intercept, c='b', label='Regr. Lineare')
plt.scatter(R_wi, I_wi)
plt.savefig(r'CurvaRI50mua', dpi=125)
plt.show()"""

def fit_lineare(I_wi, R_wi, err_I_wi=None, err_R_wi=None):
    """Esegue il fit lineare di R_wi (y) rispetto a I_wi (x) e calcola la matrice di covarianza."""
    # Calcolo della regressione lineare
    slope, intercept, r_value, p_value, std_err = linregress(I_wi, R_wi)

    # Stampa dei risultati della regressione
    print(f"Slope: {slope}, Intercept: {intercept}")

    # Calcolo della varianza residua
    n = len(I_wi)
    residuals = R_wi - (slope * I_wi + intercept)
    residual_variance = np.sum(residuals**2) / (n - 2)

    # Calcolo della varianza di slope e intercetta
    mean_I_wi = np.mean(I_wi)
    var_slope = residual_variance / np.sum((I_wi - mean_I_wi)**2)
    var_intercept = residual_variance * (1 / n + (mean_I_wi**2) / np.sum((I_wi - mean_I_wi)**2))

    # Covarianza tra slope e intercetta
    cov_slope_intercept = -mean_I_wi * var_slope / n

    # Matrice di covarianza
    cov_matrix = np.array([[var_slope, cov_slope_intercept],
                           [cov_slope_intercept, var_intercept]])

    # Aggiunta degli errori associati
    if err_I_wi is not None:
        cov_matrix[0, 0] += np.var(err_I_wi, ddof=1)
    if err_R_wi is not None:
        cov_matrix[1, 1] += np.var(err_R_wi, ddof=1)

    # Stampa dei risultati finali
    print("Matrice di covarianza:")
    print(cov_matrix)

    return slope, intercept, cov_matrix

# il parametro limiti è importante per quelle funzioni
# incui i parametri hanno dei vincoli:
# limiti = ([cmin,bmin,amin],[cmax,bmax,amax])
def fit_curve(X,Y,f,A,B,err_X=None,err_Y=None,limiti=([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])):
    params,cov_matrix = curve_fit(f,X,Y,sigma=err_Y,bounds=limiti)
    [c,b,a] = params

    Xfit = np.linspace(A,B,PRECISIONE)
    Yfit = f(Xfit,c,b,a)

    print(cov_matrix)

    return Xfit,Yfit,c,b,a,cov_matrix


def fit_polinomiale(X,Y,n):
    """Fit polinomiale di grado n sui set di dati X,Y"""
    z = np.polyfit(X,Y,n)
    p = np.poly1d(z)

    Xfit = np.linspace(min(X),max(X),PRECISIONE)
    Yfit = p(Xfit)

    return Xfit,Yfit

def find_roots(X1,Y1,X2,Y2,a,b,order=6,precision=0.0001):
    """Trova numericamente le intersezioni dei due set di dati [X1,Y1]
    e [X2,Y2] nell'intervallo [a,b] tramite un fit polinomiale all'orinde order
    (6 di default) con precisione precisione (0.001 di default)"""
    valori = []
    differenze = []

    z1 = np.polyfit(X1, Y1,order)
    p1 = np.poly1d(z1)
    Xfit1 = np.linspace(a,b,PRECISIONE)
    Yfit1 = p1(Xfit1)
    plt.plot(Xfit1,Yfit1)

    z2 = np.polyfit(X2,Y2,order)
    p2 = np.poly1d(z2)
    Xfit2 = np.linspace(a,b,PRECISIONE)
    Yfit2 = p2(Xfit2)
    plt.plot(Xfit2,Yfit2)
     
    RICERCA = np.linspace(a, b, PRECISIONE)
    for x in RICERCA:
        if np.abs(p1(x) - p2(x)) <= precision:
            valori.append(x)
            differenze.append(np.abs(p1(x) - p2(x)))

    return valori,differenze


# Propagazione dell'errore statistico nel punto (x0, y0) per una funzione y=f(x,y),
# in cui COV(x,y) = 0
def prop_no_cor(f, x0, y0, incx, incy):
    x, y = sp.symbols('x y')
    f_sym = f(x, y)  # Costruisci la funzione simbolica

    # Calcolo delle derivate parziali
    df_dx = sp.diff(f_sym, x)
    df_dy = sp.diff(f_sym, y)

    # Calcolo del valore numerico delle derivate in (x0, y0)
    df_dx_val = df_dx.evalf(subs={x: x0, y: y0})  # Derivata parziale rispetto a x in x0
    df_dy_val = df_dy.evalf(subs={x: x0, y: y0})  # Derivata parziale rispetto a y in y0

    # Somma in quadratura
    DF = sp.sqrt((df_dx_val * incx)**2 + (df_dy_val * incy)**2)

    return float(DF)


def Cov(X,Y,f):
    N = len(X)

    MX = 0
    for i in X:
        sum += i
    MX = sum/N

    MY = 0
    for i in Y:
        sum += i
    MY = sum/N
    
    SXY = 0
    for i in range(len(X)):
        SXY += (X[i] - MX)*(Y[i] - MY)
    SXY = SXY/N

    return SXY

# Propagazione dell'errore statistico nel punto (x0, y0) per una funzione y=f(x,y),
# in cui COV(x,y) non è 0.
# X e Y sono la nostra popolazione
def prop_yes_cor(f,x0,y0,incx,incy,X,Y):
    x, y = sp.symbols('x y')
    f_sym = f(x, y)  # Costruisci la funzione simbolica

    # Calcolo delle derivate parziali
    df_dx = sp.diff(f_sym, x)
    df_dy = sp.diff(f_sym, y)

    # Calcolo del valore numerico delle derivate in (x0, y0)
    df_dx_val = df_dx.evalf(subs={x: x0, y: y0})  # Derivata parziale rispetto a x in x0
    df_dy_val = df_dy.evalf(subs={x: x0, y: y0})  # Derivata parziale rispetto a y in y0

    return prop_no_cor(f,x0,y0,incx,incy) + 2*df_dx_val*df_dy_val*Cov(X,Y)
 

def consistenza_statistica(X,Y,incX,incY):
    t = np.abs(Y-X)/np.sqrt(incX**2 + incY**2)

    a = ((norm.cdf(t))*100)/68.27 # WOW!!!
    print("a: ",a)
    if(a > 0.005):
        return True,a
    elif(a < 0.0003):
        return False,a
    else:
        return True,a

def f(x,y):
    return 