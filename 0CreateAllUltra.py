
# %% codecell
'PACKAGES'
from numba import jit, prange
import concurrent.futures
import numpy as np
import cmath
import math
import pickle
import time
import os
import platform
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
plt.ion()
epsilon=sys.float_info.epsilon

# %% codecell
######################################################
#################### ALL FUNCTIONS ###################
######################################################
#'DEF FREE POLYGONAL CURVE'
def fFree(p):
    CurvaJordan=plt.figure(figsize=(8,8))
    curvadi=np.array([complex(1.8,0)])
    plt.title('Choose the vertices of your Jordan curve')
    A=str(p-1)
    plt.xlabel('You have ' + A +' vertices left')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(curvadi[0].real,curvadi[0].imag,'ro')
    plt.draw()

    i = 1;
    for i in range (1,p):
        plt.plot(1.8,0,'ro')
        inpto=plt.ginput(1)
        pto=complex(inpto[0][0],inpto[0][1])
        plt.plot(pto.real,pto.imag,'ro');
        B=str(p-i-1)
        plt.xlabel('You have ' + B +' vertices left')
        curvadi=np.append(curvadi,pto)
        plt.plot([curvadi[i-1].real,curvadi[i].real],[curvadi[i-1].imag,curvadi[i].imag]);
        plt.draw()

    curvadi=np.append(curvadi,complex(1.8,0))
    plt.plot([curvadi[p-1].real,curvadi[p].real],[curvadi[p-1].imag,curvadi[p].imag]);
    plt.draw()
    return [curvadi, CurvaJordan]

#'DEF CIRCLE (REGULAR POLYGON)'
def fCircle(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    circulo=[]
    for i in range (0,p+1):
        ci=cmath.rect(1.5,(2*(np.pi)*i+np.pi)/p)
        circulo=np.append(circulo, ci)
    plt.plot(circulo.real,circulo.imag, marker='o', ms=5, mec='red', mfc='red', c='cyan')
    plt.draw()
    return [circulo, CurvaJordan]

#'DEF ELLIPSE'
def fEllipse(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    elipse=[]
    for i in range (0,p+1):
        el=complex((1.8)*np.cos((2*(np.pi)*i)/p), (1)*np.sin((2*(np.pi)*i)/p))
        elipse=np.append(elipse, el)
    plt.plot(elipse.real,elipse.imag, marker='o', mec='red', ms=5, mfc='red', c='cyan')
    plt.draw()
    return [elipse, CurvaJordan]

#' DEF CARDIOIDE'
def fCardioid(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    cardioide=[]
    for i in range (0,p+1):
        car=cmath.rect(1-1*(np.sin((2*(np.pi)*i)/p)), (2*(np.pi)*i)/p)
        cardioide=np.append(cardioide, car+(0+1j))
    plt.plot(cardioide.real,cardioide.imag, marker='o',ms=5, mec='red', mfc='red', c='cyan')
    plt.draw()
    return [cardioide, CurvaJordan]

#'DEF FLOWER'
def fFlower(p,pet):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')
    trebol=[]
    for i in range (0,p+1):
        tre=(0.5)*(cmath.rect(2+(1)*(np.cos((pet)*(2*(np.pi)*i)/p)), (2*(np.pi)*i)/p))
        trebol=np.append(trebol, tre)
    plt.plot(trebol.real,trebol.imag, marker='o', ms=5, mec='red', mfc='red', c='cyan')
    plt.draw()
    return [trebol, CurvaJordan]

#'DEF BUTTERFLY'
def fButterfly(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    mariposa=[]
    for i in range (0,p+1):
        mar=cmath.rect(1-(np.cos((2*(np.pi)*i)/p))*(np.sin(3*(2*(np.pi)*i)/p)), (2*(np.pi)*i)/p)
        mariposa=np.append(mariposa, mar)
    plt.plot(mariposa.real,mariposa.imag, marker='o', ms=5, mec='r', mfc='r', c='c')
    plt.draw()
    return [mariposa, CurvaJordan]

#'DEF SPIRAL'
def fSpiral(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    espiral=[]

    for i in range (1,(p+1)//2):
        es=cmath.rect((0.8)*(((np.pi)*i)/((p))),(9*(np.pi)*i)/((p)))
        espiral=np.append(espiral, es)
    for i in range (((p+1)//2), p):
        pir=cmath.rect((1)*((np.pi)*(p-i))/((p)),(9*(np.pi)*((p-i))/(p)))
        espiral=np.append(espiral, pir)

    espiral=np.append(espiral, espiral[0])
    plt.plot(espiral.real,espiral.imag, marker='o', ms=5, mec='red', mfc='red', c='cyan')
    plt.plot([espiral[0].real,espiral[p-2].real],[espiral[0].imag,espiral[p-2].imag],marker='o', ms=5, mec='red', mfc='red', c='cyan');
    plt.draw()
    return [espiral, CurvaJordan]
#Ojo: este espiral tiene un vertice menos de los que usted eligio


#'DEF DRAGON'
def fDragon(p):
    CurvaJordan=plt.figure(figsize=(3,3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('This is how your Jordan curve looks like')

    vibora=[]
    jor=7
    gor=0.5
    for i in range (0, p//2):
        vi = complex( 3*i/(p//2)-(1.5), (1)*np.sin(jor*(3*i/(p//2))) +gor )
        vibora=np.append(vibora, vi)

    i= p//2
    ita=complex(3*(p-i)/(p//2)-(1.5), (1)*np.sin(jor*(3*(p-i)/(p//2)))-gor )
    arti1=(0.5)*(ita+vibora[(p//2)-1])+complex(0.3,0)
    vibora=np.append(vibora, arti1)

    for i in range ((p//2), p):
        bor = complex(3*(p-i)/(p//2)-(1.5), (1)*np.sin(jor*(3*(p-i)/(p//2)))-gor )
        vibora=np.append(vibora, bor)
    arti2=complex(-1.8, 0)
    vibora=np.append(vibora, arti2)
    vibora=np.append(vibora,vibora[0])

    plt.plot(vibora.real,vibora.imag, marker='o', ms=5, mec='red', mfc='red', c='cyan')
    plt.draw()
    return [vibora, CurvaJordan]
#Ojo: esta viborita tiene dos vertices mas de los que usted eligio

#DEF ZIG-ZAG FRACTALISING
def ZigZag(I,fc,curvafe,StoreCurvas,CurvaJordan):
    CurvaIZ=CurvaJordan
    i=1
    while i<=I:
        m=len(curvafe)
        cufe=curvafe[0]
        for k in range (0,m-1):
            wf=(curvafe[k+1]+curvafe[k])*(0.5)
            z1f=(curvafe[k+1]-curvafe[k])*fc*(0.5)
            z2f=complex(z1f.imag,-z1f.real)+wf
            z3f=complex(-z1f.imag,z1f.real)+wf
            cufe=np.append(cufe,[z2f,z3f,curvafe[k+1]])
        StoreCurvas=StoreCurvas+[cufe]
        CurvaIZ=plt.figure(figsize=(4,4))
        curvafe=cufe
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(cufe.real,cufe.imag)
        plt.draw()
        i += 1

    CurvaJordan=CurvaIZ
    return [StoreCurvas, CurvaJordan]
# Dato curioso: la i-esima curva iterada tiene
#(p*(3**(i+1)-1)//2+i+1)-(p*(3**i-1)//2+i)=p3**i+1 entradas.

#DEF SNOWFLAKE FRACTALISING
def Triangulito(I,fc,curvafe,StoreCurvas,CurvaJordan):
    CurvaIT=CurvaJordan
    i=1
    while i<=I:
        m=len(curvafe)
        cufe=curvafe[0]
        for k in range (0,m-1):
            wf=(curvafe[k+1]+curvafe[k])*(0.5)
            w1f=(curvafe[k+1]-curvafe[k])*(fc)
            z2f=complex(w1f.imag,-w1f.real)*(0.5)*(np.sqrt(3))+wf
            z1f=wf-w1f*(0.5)
            z3f=wf+w1f*(0.5)
            cufe=np.append(cufe,[z1f,z2f,z3f,curvafe[k+1]])
        StoreCurvas=StoreCurvas+[cufe]
        CurvaIT=plt.figure(figsize=(4,4))
        curvafe=cufe
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(cufe.real,cufe.imag)
        plt.draw()
        i += 1

    CurvaJordan=CurvaIT
    return [StoreCurvas,CurvaJordan]
# Dato curioso: la i-esima curva iterada tiene
#(p*(4**(i+1)-1)//3+i+1)-(p*(4**i-1)//3+i) vertices

##DEF SPIRALS FRACTALISING
def Espiral1(I,fc,curvafe,StoreCurvas,CurvaJordan):
    CurvaIE=CurvaJordan
    i=1
    while i<=I:
        m=len(curvafe)
        cufe=curvafe[0]
        for k in range (0,m-1):
            wf=(curvafe[k+1]+curvafe[k])*(0.5)
            z1f=(curvafe[k+1]-curvafe[k])*fc*(0.5)
            z2f=complex(z1f.imag,-z1f.real)+wf
            z3f=complex(-z1f.imag,z1f.real)+wf
            a1=(curvafe[k]+wf)*(0.5)
            a2=(z2f+wf)*(0.5)
            a3=(curvafe[k+1]+wf)*(0.5)
            a4=(z3f+wf)*(0.5)
            b1=(a1+wf)*(0.5)
            b2=(a2+wf)*(0.5)
            b3=(a3+wf)*(0.5)
            cufe=np.append(cufe,[z2f,a3,a4,b1,b2,wf,b3,a2,a1,z3f,curvafe[k+1]])
        StoreCurvas=StoreCurvas+[cufe]
        curvafe=cufe
        CurvaIE=plt.figure(figsize=(4,4))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(cufe.real,cufe.imag)
        plt.draw()
        i += 1

    CurvaJordan=CurvaIE
    return [StoreCurvas, CurvaJordan]

#DEF DECREASING SPIRALS FRACTALISING
def Espiral2(I,fc,curvafe,StoreCurvas,CurvaJordan):
    CurvaIE2=CurvaJordan
    i=1
    while i<=I:
        m=len(curvafe)
        cufe=curvafe[0]
        for k in range (0,m-1):
            espia=[curvafe[k],curvafe[k+1]]
            StorEspia=[]
            j1=1
            while j1<=5:
                wf=(curvafe[k]+curvafe[k+1])*(0.5)
                z1f=((-1)**(j1)*espia[1]+(-1)**(j1+1)*espia[0])*fc*(0.8)
                z2f=complex(z1f.imag,-z1f.real)+wf
                z3f=complex(-z1f.imag,z1f.real)+wf
                StorEspia=np.append(StorEspia,[z2f,z3f])
                espia=[z2f,z3f]
                j1+=1
            cufe=np.append(cufe,[StorEspia[0],StorEspia[3],StorEspia[5],StorEspia[6],StorEspia[8],wf,StorEspia[7],StorEspia[4],StorEspia[2],StorEspia[1],curvafe[k+1]])
        StoreCurvas=StoreCurvas+[cufe]
        curvafe=cufe
        CurvaIE2=plt.figure(figsize=(4,4))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(cufe.real,cufe.imag)
        plt.draw()
        i += 1

    CurvaJordan=CurvaIE2
    return [StoreCurvas, CurvaJordan]

# Tow segments intersection function
@jit(nopython=True,nogil=True)
def SegInter(s11,s12,s21,s22):
    S1=s12-s11
    S1T=np.array([S1[1],-S1[0]])
    S2=s22-s21
    PrS2=np.dot(S2,S1T)
    if PrS2==0:
        return np.nan
    SS=s11-s21
    PrSS=np.dot(SS,S1T)
    PrS2=np.dot(S2,S1T)
    xs=(PrSS/PrS2)
    if xs<0 or xs>1:
        return np.nan
    x12=S2*(PrSS/PrS2)+s21
    return np.complex(x12[0],x12[1])

############################################
####DEFINITION f_GAMMA R3 AND MOEBIUS 2D####
############################################

#Intersections of the Jordan Curve with a single line segment + translation for Moebius 2D
@jit(nopython=True,nogil=True)
def CurveLineInters(curve:np.ndarray,lsp1:np.complex128,lsp2:np.complex128, epsilon):
    numv=curve.size
    g=np.empty(0,dtype=np.complex128)
    gm=np.empty(0,dtype=np.complex128)
    for i in range (1, numv):
        p1=np.array([curve[i-1].real,curve[i-1].imag])
        p2=np.array([curve[i].real,curve[i].imag])
        InterPoint=SegInter(lsp1,lsp2,p1,p2)
        if np.isnan(InterPoint)==True:
            pass
        else:
            if abs(p1[0]-p2[0])<epsilon:
                t=(InterPoint.imag-p2[1])/(p1[1]-p2[1])
            else:
                t=(InterPoint.real-p2[0])/(p1[0]-p2[0])
            c1=cmath.rect(1,(1-t)*((2*(np.pi)*i)/(numv-1) ) + (t)*(2*(np.pi)*(i-1)/(numv-1)))
            g=np.append(g,InterPoint)
            gm=np.append(gm,c1)
    return [g,gm]

# coordinates of f_gamma(a,b) for all a,b in a single intersecting line (for R3 & Moebius 2D)
@jit(nopython=True,nogil=True)
def Coords1Line(g,gm,f_g1dir,f_g1dirM):
    i1=1
    while i1 <= g.size:
        j1=1
        while  j1+i1  <= g.size:
            #calculo de puntos en R3
            h = 0.5*( g[i1-1]+g[i1-1+j1] )
            nm = abs(g[i1-1]-g[i1-1+j1])
            newpt=np.array([h,nm],dtype=np.complex128)
            f_g1dir=np.append(f_g1dir,newpt)
            #calculo de puntos en la banda
            ham = np.angle(gm[i1-1]+gm[i1-1+j1],deg=False)
            hm = cmath.rect(1,ham)
            nmm = abs(gm[i1-1]-gm[i1-1+j1])
            hm=(1+nmm)*hm
            newptm=np.array([hm,0],dtype=np.complex128)
            f_g1dirM=np.append(f_g1dirM,newptm)
            j1 +=1
        i1 +=1
    return [f_g1dir,f_g1dirM]

#Generate grid and append all calculations for one fixed direction
@jit(nopython=True,nogil=True)
def GridLoop(grid,degree,curvar):
    f_g1dir=np.empty(0,dtype=np.complex128)
    f_g1dirM=np.empty(0,dtype=np.complex128)
    angulo = (degree)*(np.pi)/180
    dir=cmath.rect(1,angulo)
    rot=cmath.rect(1,angulo+np.pi/2)
    rotgrid=np.multiply(grid,rot)
    for k in range (0,grid.size):
        s1p1=np.array([(rotgrid[k]+3*dir).real,(rotgrid[k]+3*dir).imag])
        s1p2=np.array([(rotgrid[k]-3*dir).real,(rotgrid[k]-3*dir).imag])
        G=CurveLineInters(curvar,s1p1,s1p2, epsilon)
        g=G[0]
        gm=G[1]
        F_G=Coords1Line(g,gm,f_g1dir,f_g1dirM)
        f_g1dir=F_G[0]
        f_g1dirM=F_G[1]
    return [f_g1dir,f_g1dirM]

def GridLoopPack(listpack):
    return GridLoop(*listpack)

###COORDINATES F_GAMMA R3 & MOEBIUS 2D FOR ALL ANGLES###
def f_gamma(curvar, r, deg, noa, pasito):
    start = time.time()
    f_g90=[]
    f_g90M=[]
    b=np.arange(-3, 3, r, dtype=np.complex128)
    angulitos=np.arange(deg,deg+noa, pasito)
    numang=angulitos.size
    angulitospack=[]
    for k in range (0,numang):
        angulitospack.insert(k,[b,angulitos[k],curvar])
    with concurrent.futures.ProcessPoolExecutor(workingcores) as executor:
        AllAng=list(tqdm(executor.map(GridLoopPack, angulitospack),total=numang))
    for ThisDir in AllAng:
        f_g1dir=ThisDir[0]
        f_g1dirM=ThisDir[1]
        f_g90=f_g90+[f_g1dir]
        f_g90M=f_g90M+[f_g1dirM]
    end = time.time()
    print('\nThis step took '+str(time.gmtime(end-start)[7]-1)+ ' days, ' +str(time.gmtime(end-start)[3])+ ' hours, ' +str(time.gmtime(end-start)[4])+ ' minutes and ' +str(time.gmtime(end-start)[5])+ ' seconds', flush=True)
    return [f_g90,f_g90M]

#####################################
####DEFINITION F_GAMMA MOEBIUS 3D####
#####################################


#Calculate the Moebius 3D coordinates for a single given direction
@jit(nopython=True,nogil=True)
def f_gM3D1Dir(i,moebius2Di):
    f_gM3D=np.empty(0,dtype=np.complex128)
    for k in np.arange(0,(moebius2Di).size//2):
        z1 = moebius2Di[2*k]
        if cmath.phase(z1)>0:
            z=z1
            z = (z**2)/(abs(z))
            if np.angle(z)>=0:
                teta = np.angle(z)
            else:
                teta= 2*np.pi+cmath.phase(z)
            v = cmath.rect(1, -teta/2 + np.pi/2)
            w = (3-abs(z))*v
            z = cmath.rect(3,teta)+cmath.rect(w.real, teta)
            zh = complex(w.imag,0)
            f_gM3D = np.append(f_gM3D, [z,zh])
        if cmath.phase(z1) <= 0:
            z=z1
            z = z*cmath.rect(1,np.pi)
            z = (z**2)/(abs(z))
            if np.angle(z)>=0:
                teta = np.angle(z)
            else:
                teta= 2*np.pi+cmath.phase(z)
            v = cmath.rect(1, -teta/2 + 3*np.pi/2)
            w = (3-abs(z))*v
            z = cmath.rect(3,teta)+cmath.rect(w.real,teta)
            zh = complex(w.imag,0)
            f_gM3D = np.append(f_gM3D, [z, zh])
    return f_gM3D

def f_gM3D1DirPack(listpack):
    return f_gM3D1Dir(*listpack)

# Coordinates of f_gamma Moebius 3D for all directions
def f_gammaM3D(moebius2D):
    start=time.time()
    numang=len(moebius2D)
    Ang2Dpack=[]
    for k in range (0,numang):
        Ang2Dpack.insert(k,[k,moebius2D[k]])
    with concurrent.futures.ProcessPoolExecutor(workingcores) as executor:
        f_gM3DT=list(tqdm(executor.map(f_gM3D1DirPack, Ang2Dpack),total=numang))
    end = time.time()
    print('\nThis step took '+str(time.gmtime(end-start)[7]-1)+ ' days, ' +str(time.gmtime(end-start)[3])+ ' hours, ' +str(time.gmtime(end-start)[4])+ ' minutes and ' +str(time.gmtime(end-start)[5])+ ' seconds', flush=True)
    return f_gM3DT

#%%codecell
######################################################
#################### MAIN PROGRAM ####################
######################################################

#GENERATE JORDAN CURVE
if __name__== "__main__":
    print("\n **WELCOME TO CHASING SQUARES!** \nCopyright 2021, Ulises Morales-Fuentes & Cristina Villanueva-Segovia, all rights reserved. \nYou are about to create a Jordan curve. ")
    p_in= int(input('First choose the number of vertices: '))
    AllTypesCurves=[fFree, fCircle, fEllipse, fCardioid, fFlower, fButterfly, fSpiral, fDragon]
    sc = int(input('Which type of curve do you want to start with? we have:\n 0. Free Polygonal Curve\n 1. Circle (Regular Polygon)\n 2. Ellipse\n 3. Cardioid\n 4. Flower\n 5. Butterfly\n 6. Spiral\n 7. Dragon\nJust type the number of your choice and press return: ' ))
    if sc==4:
        pet_in=int(input('How many petals would you like your flower to have?: '))
        out=AllTypesCurves[sc](p_in,pet_in)
        curvafe_in= out[0]
        CurvaJordan_in=out[1]
    else:
        out=AllTypesCurves[sc](p_in)
        curvafe_in=out[0]
        CurvaJordan_in=out[1]

    StoreCurvas_in=[curvafe_in]
    AllTypesFract=[ZigZag,Triangulito, Espiral1, Espiral2]
    BoolFractProc=[False]*4
    fp=int(input('Would you like to run a \'fractalising\' process on your curve?. You can choose between:\n 0. None\n 1. Zig-Zag\n 2. Snowflake\n 3. Spirals\n 4. Decreasing Spirals\n Just type the number of your choice and press return: ' ))
    if fp==0:
        curvar_in=StoreCurvas_in[0]
        I_in=0
        fc_in='N/A'
    else:
        I_in=int(input('How many times would you like the \'fractalising\' process to be runned? (type an integer number): '))
        fc_in=float(input('From 0 to 1, how far would you like to allow the \'fractalised\' curve to be from the original curve? : '))
        out=AllTypesFract[fp-1](I_in,fc_in,curvafe_in,StoreCurvas_in,CurvaJordan_in)
        curvar_in=out[0][I_in]
        CurvaJordan_in=out[1]
        BoolFractProc[fp-1]=True

    #%%codecell
    print("OK! Your Jordan Curve has been created!\nNow, to analyse the behavour of the inscribed square(s), choose the following paramaters:")
    r_in= float(input('Choose the resolution of the set f_gamma, this is the distance between points\n(the smaller, the better but the longer it takes): '))
    deg_in =int(input('Fix an initial angle in degrees: '))
    noa_in=int(input('Fix a final angle in degrees: '))-deg_in
    pasito_in=float(input('Choose a step size to go from ' +str(deg_in)+' to ' +str(deg_in+noa_in)+': '))
    ThreeDCh=str(input("Do you want to calculate the f_gamma coordinates for the 3 dimensional Moebius strip animation? (Y/N): "))
    while ThreeDCh not in ['y','n','Y','N','yes','no','YES','NO','Yes','No']:
        ThreeDCh=str(input("I beg you pardon! That\'s not an option, try again please: "))
    print("Finally, let\'s create a directory to save the information... \n")

    # %% codecell
    #CREAR NUEVO DIRECTORIO PARA GUARDAR LOS DATOS DE LAS CURVAS
    cwd = os.getcwd()
    print("You are currently in this directory: " +cwd)
    #the next path locates the directory where all curves are gonna be saved always
    currentOS = platform.system()
    cho=int(input('In which computer are we working right now?:\n For Ulises\' PC type 0 \n For Cristina\'s PC type 1\n For any other cumputer type 2:\n '))
    WinCristinaPath = os.path.join('C:\\','Users', 'crivi', 'OneDrive', 'Desktop', 'CurvasIpy', 'PickledCurves')
    LinCristinaPath = os.path.join('/home', 'cristina', 'Documents', 'CurvasIPy', 'PickledCurves' )
    WinUlisesPath = os.path.join('C:\\','Users', 'ulise', 'Documents', 'ChasingSquares', 'PickledCurves')
    LinUlisesPath = os.path.join('/home', 'ulises', 'Documents', 'PickledCurves' )
    while cho not in [0,1,2]:
        cho=int(input("I beg you pardon! That\'s not an option, try again please: "))
    if cho==0:
        if currentOS == 'Windows':
            ChosenPath = WinUlisesPath
        elif currentOS == 'Linux':
            ChosenPath = LinUlisesPath
    elif cho==1:
        if currentOS == 'Windows':
            ChosenPath = WinCristinaPath
        elif currentOS == 'Linux':
            ChosenPath = LinCristinaPath
    elif cho==2:
        ChosenPath = cwd

    if os.path.exists(ChosenPath):
        print(ChosenPath + ' : exists')
        if os.path.isdir(ChosenPath):
            print(ChosenPath+ ' : is a directory, I will create a new directory in this directory')

    #create new directory
    NewCurve=str(input('How would you like this new directory to be named?: '))
    dir = os.path.join(ChosenPath, NewCurve)

    while os.path.exists(dir)==True:
        NewCurve=str(input("THE DIRECTORY '" +NewCurve+ "' ALREADY EXISTS, PLEASE TRY ANOTHER NAME: "))
        dir = os.path.join(ChosenPath, NewCurve)
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Directory '" +NewCurve+ "' created")

    # %% codecell
    cores=os.cpu_count()
    workingcores=int(input("The calculations will start now; how many cores would you like to use for runing these calculations? (you have a total of "+str(cores)+"): "))
    print("OK!, data will be automatically saved in the '" +NewCurve+ "' directory and will be ready for visualisation.\n")
    print('We are now working on the R3 & Moebius 2D calculation. This will run in two steps : \nStep 1 running...')
    OUTFgammaR = f_gamma(curvar_in,r_in,deg_in,noa_in,pasito_in)
    roja90=OUTFgammaR[0]
    roja90m=OUTFgammaR[1]
    print('Step 2 running...')
    OUTFgammaA = f_gamma(curvar_in,r_in,deg_in+90,noa_in,pasito_in)
    azul180=OUTFgammaA[0]
    azul180m=OUTFgammaA[1]

    # %% codecell
    #GUARDAR ARCHIVOS PICKLE EN EL NUEVO DIRECTORIO

    #Crear direcciones de archivos en el nuevo directorio
    dirRoja = os.path.join(ChosenPath, NewCurve,'roja90Pickle' )
    dirAzul = os.path.join(ChosenPath, NewCurve,'azul180Pickle' )
    dirRojaM = os.path.join(ChosenPath, NewCurve,'roja90mPickle' )
    dirAzulM = os.path.join(ChosenPath, NewCurve,'azul180mPickle')
    dirCurvar = os.path.join(ChosenPath, NewCurve,'curvarPickle')
    dirText = os.path.join(ChosenPath, NewCurve,'TextData.txt')
    dirAngIn = os.path.join(ChosenPath, NewCurve,'angin.txt')
    dirAngFi = os.path.join(ChosenPath, NewCurve,'angfi.txt')
    dirPasito = os.path.join(ChosenPath, NewCurve,'pasito.txt')
    dirResol= os.path.join(ChosenPath, NewCurve,'resolution.txt')
    dirImagen = os.path.join(ChosenPath, NewCurve,'CurvaJordan.png')
    #Guardar curvas en las direcciones creadas
    CreateRoja90 = open(dirRoja, "wb")
    pickle.dump(roja90, CreateRoja90)
    CreateRoja90.close()

    CreateAzul180 = open(dirAzul, "wb")
    pickle.dump(azul180, CreateAzul180)
    CreateAzul180.close()

    CreateRoja90M = open(dirRojaM, "wb")
    pickle.dump(roja90m, CreateRoja90M)
    CreateRoja90M.close()

    CreateAzul180M = open(dirAzulM, "wb")
    pickle.dump(azul180m, CreateAzul180M)
    CreateAzul180M.close()

    CreateCurvar = open(dirCurvar, "wb")
    pickle.dump(curvar_in, CreateCurvar)
    CreateCurvar.close()

    CreateAngIn = open(dirAngIn, "wb")
    pickle.dump(deg_in, CreateAngIn)
    CreateAngIn.close()

    CreateAngFi = open(dirAngFi, "wb")
    pickle.dump(noa_in, CreateAngFi)
    CreateAngFi.close()

    CreatePasito = open(dirPasito, "wb")
    pickle.dump(pasito_in, CreatePasito)
    CreatePasito.close()

    CreateResol = open(dirResol, "wb")
    pickle.dump(r_in, CreateResol)
    CreateResol.close()

    CreateText = open(dirText, "w+")
    CreateText.write("**DATA OF THE CURVE \'" +NewCurve+ "\'" "**\n"
                     "\n1.- Initial number of vertices= " +str(p_in)+ "\n"
                     "2.- Type of function used for base jordan curve= "+str(AllTypesCurves[sc])+"\n"
                     "3.- Type of fractalising process: Zig-Zag= " +str(BoolFractProc[0])+ " Snowflake= " +str(BoolFractProc[1])+  ", \nSpirals= " +str(BoolFractProc[2])+ ", Decreasing Spirals= "+str(BoolFractProc[3])+"\n"
                     "4.- Number of \'fractalising\' iterations= " +str(I_in)+ "\n"
                     "5.- Proximity factor for \'fractalising\' iterations= " +str(fc_in)+ "\n"
                     "6.- Final number of vertices= " +str(len(curvar_in))+ "\n"
                     "7.- Resolution of the f_gamma set= " +str(r_in)+ "\n"
                     "8.- Initial angle= " +str(deg_in)+"\n"
                     "9.- Final angle= "+str(noa_in+deg_in)+"\n"
                     "10.- Step between angles= "+str(pasito_in)+"\n"
                    )
    CreateText.close()

    CurvaJordan_in.savefig(dirImagen)
# Run moebius 3D if decided
    if ThreeDCh in ['y','Y','yes','YES','Yes']:
        print('\nWe are now working on the Moebius 3D calculation. This will run in  2 steps: \nStep 1 running...')
        roja90m3d=f_gammaM3D(roja90m)
        print('Step 2 running...')
        azul180m3d=f_gammaM3D(azul180m)

    #Save Moebius 3D files to directory
        dirRojaM3D = os.path.join(ChosenPath, NewCurve,'roja90m3DPickle' )
        dirAzulM3D = os.path.join(ChosenPath, NewCurve,'azul180m3DPickle')

        CreateRoja90M3D = open(dirRojaM3D, "wb")
        pickle.dump(roja90m3d, CreateRoja90M3D)
        CreateRoja90M3D.close()

        CreateAzul180M3D = open(dirAzulM3D, "wb")
        pickle.dump(azul180m3d, CreateAzul180M3D)
        CreateAzul180M3D.close()

    if os.path.exists(dirRoja) and (os.path.exists(dirAzulM) and os.path.exists(dirCurvar)):
        print("\nAll done! The directory '" +NewCurve+ "' has been created and files are being loaded" )

    #%%codecell
