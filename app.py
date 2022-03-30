# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import random
import skfuzzy as fuzz
from skimage import data
from skimage  import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian
import matplotlib.pyplot as plt

 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/' 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def HEfun(Ruta):
    RUTA=str(Ruta)
    gris=cv2.imread(RUTA,0)
    histo=np.zeros(256)
    filas=gris.shape[0] #
    columnas=gris.shape[1]
    salida=np.zeros((filas,columnas))
    for i in range(filas):
        for j in range(columnas):
            pixel = int(gris[i,j])
            histo[pixel]  +=1   #ocurrencia en el univero
    pro = histo/(filas*columnas)  
    ecualiza=np.zeros(256)
    acumulado = 0
    for k in range(256):
        acumulado = pro[k] + acumulado
        ecualiza[k]=acumulado * 255.0           
    for i in range(filas):
        for j in range(columnas):
            entrada = int(gris[i,j])
            salida[i,j]=ecualiza[entrada]
    return salida
def HEBMpfun(Ruta):
    RUTA=str(Ruta)
    im=cv2.imread(RUTA,0)
    a,b = im.shape
    n_i=np.zeros((256))
    y=np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            pxl=im[i,j]#Valor del pixel en la posicion (i,j)
            n_i[pxl]+=1
    pro=n_i/(a*b)
    ecualiza=np.zeros(256)
    acumulado = 0
    for k in range(256):
            acumulado = pro[k] + acumulado
            ecualiza[k]=acumulado 
    #Treshold
    #G=240
    G=int(request.form.get("G"))
    #####
    c_i=np.zeros((256))
    for i in range(a):
        for j in range(b):
            pxl=im[i,j]#Valor del pixel en la posicion (i,j)
            if pxl<=G:
                c_i[pxl]+=1
    for i in range(256):
        if c_i[i]>0:
            c_i[i]=1/c_i[i]
    L_k=np.zeros(256)
    acumulado = 0
    for k in range(256):
            acumulado = c_i[k] + acumulado
            L_k[k]=acumulado 
    N_c=sum(c_i)
    G_g=255/N_c
    L_k=L_k*G_g
    for i in range(a):
        for j in range(b):
           entrada =im[i,j]
           y[i,j]=L_k[entrada]
    return y
def FUZZYfun(Ruta):
    RUTA=str(Ruta)
    pixel = np.linspace(0, 255, 256)
    claros = fuzz.smf(pixel, 130, 230)
    grises = fuzz.gbellmf(pixel, 55, 3, 128)
    oscuros = fuzz.zmf(pixel, 25, 130)
##################################################
    # s1 = 30
    # s2 = 40
    # s3 = 245
    s1=int(request.form.get("s1"))
    s2=int(request.form.get("s2"))
    s3=int(request.form.get("s3"))
    salida = np.zeros(256)
    for i in range (256):
        salida [i] = ((oscuros[i]*s1)+(grises[i]*s2)+(claros[i]*s3)) / (oscuros[i]+grises[i]+claros[i])
    
    #=======================================================================#
    gris = cv2.imread(RUTA,0)
    [filas, columnas] = gris.shape
    EHF = np.zeros((filas, columnas))
    
    for i in range(filas):
        for j in range(columnas):
            valor = gris[i, j]
            EHF[i,j] = np.uint8(salida[valor])
    return EHF


def Unsharpfun(Ruta):
    RUTA=str(Ruta)    
    kernel = np.array([
      [-1, -1, -1],
      [-1, 9, -1],
      [-1, -1, -1]
    ])
    imagen=cv2.imread(RUTA,0)
    img3=cv2.filter2D(imagen, -1, kernel)
    return img3

def Linealfun(Ruta):
    RUTA=str(Ruta)
    gris=cv2.imread(RUTA,0)

    r1=int(request.form.get("r1"))
    r2=int(request.form.get("r2"))
    t1=int(request.form.get("t1"))
    t2=int(request.form.get("t2"))

    #Cálculo de las pendientes
    m1=t1/r1;
    m2=(t2-t1)/(r2-r1);
    m3=(255-t2)/(255-t2);

    #Calculo de las b2,b3
    b2=t1-(m2*r1);
    b3=t2-(m3*r2);

    filas=gris.shape[0] 
    columnas=gris.shape[1]
    matriz=np.zeros((filas,columnas))
    nueva=np.zeros((filas,columnas))

    for i in range(filas):
        for j in range(columnas):
            matriz= gris[i,j]
            if matriz<=r1:
                nueva[i,j]=m1*matriz
            elif (r1<matriz & matriz<=r2):
                nueva[i,j]=m2*(matriz)+b2
            elif matriz>r2:
                nueva[i,j]=m3*(matriz)+b3   
    return nueva


@app.route('/')
def home():
    return render_template('index.html')
 
   
@app.route('/', methods=['POST'])
def upload_image1():
    if 'Archivo' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['Archivo']
    if file.filename == '':
        flash('No se seleccionó ninguna imagen para subir')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if request.form.get('v1') == 'HE':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RUTA=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #########################
            salida=HEfun(RUTA)
        
            filename2='processHE'+filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename2),salida)        
            flash('Imagen procesada de manera exitosa HE: ')
            return render_template('index.html', filename=filename2)
        elif  request.form.get('v2') == 'HEBM+':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RUTA=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #########################
            salida=HEBMpfun(RUTA)
            
            b=str(random.randrange(100))
            bH=str('processHEBM')+str(b)
            #filename3='processFUZZY'+filename 
            filename1=bH+filename
            
            #filename1='processHEBM'+filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename1),salida)        
            flash('Imagen procesada de manera exitosa hebm+: ')
            return render_template('index.html', filename=filename1)
        elif  request.form.get('v3') == 'Fuzzy':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RUTA=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #########################
            salida=FUZZYfun(RUTA)
            
            a=str(random.randrange(100))
            ab=str('processHEBM')+str(a)
            #filename3='processFUZZY'+filename 
            filename3=ab+filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename3),salida)        
            flash('Imagen procesada de manera exitosa FUZZY: ')
            return render_template('index.html', filename=filename3)
        elif  request.form.get('v4') == 'Unsharp':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RUTA=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #########################
            salida=Unsharpfun(RUTA)
        
            filename6='processUNSHARPI'+filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename6),salida)        
            flash('Imagen procesada de manera exitosa por Unsharp: ')
            return render_template('index.html', filename=filename6)
        elif  request.form.get('v5') == 'Lineal':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RUTA=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #########################
            salida=Linealfun(RUTA)
            
            c=str(random.randrange(100))
            cl=str('processLINEAL')+str(c)
            #filename3='processFUZZY'+filename 
            filename5=cl+filename
            #filename5='processLINEAL'+filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename5),salida)        
            flash('Imagen procesada de manera exitosa por Transformacion Lineal: ')
            return render_template('index.html', filename=filename5)
        else:
            pass # unknown
        
    else:
        flash('Solo imagenes con extensión: png, jpg, jpeg, gif')
        return redirect(request.url)
 
 
@app.route('/display/<filename>')
def display_image1(filename): 
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()