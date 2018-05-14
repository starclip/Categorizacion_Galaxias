# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:52:46 2018

@author: Imperio Starclip
"""

# -*- coding: utf-8 -*-
import cv2
import numpy as np
from glob import glob

cont = 0;

# Verifica el determinante del mapeo para ver sí es posible de realizar
def es_inverso(a, b, c, d):
    return True if (b*c - a*d) == 0 else False

# Calcula el valor de W en terminos de Z
def formula_W (z, a, b, c, d):
    w = ((a*z) + b) / ((c*z) + d)
    return w

# Leer imagenes y almacenarlas en un arreglo.
def obtener_imagenes(imagenes):
    i=0;
    for image in glob('DR14/*.jpg'):
        imagen = cv2.imread(image, 1);
        imagenes.append(imagen);
        i += 1;

# Función ReLu
def Relu(x):
    return x * (x > 0)
        
# Obtener las imagenes en gris.
def obtener_gris(imagenes, imagenes_gray):
    global cont;
    cont = 1;
    kernel = np.ones((5,5),np.uint8)
    for i in imagenes:
        img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY);
        blur = cv2.GaussianBlur(img_gray,(5,5),0)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        erosion = cv2.erode(th, kernel,3) 
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, (5,5))
        denoising = cv2.fastNlMeansDenoising(opening, None, 10, 21, 7)
        relu = Relu(opening)
        imagenes_gray.append(relu);
        name = "DR14-Gray/gray_image" + str(cont) + ".jpg";
        cv2.imwrite(name, relu);
        cont += 1;
        

# Mostrar las imagenes.
def ver_imagenes(imagenes, imagenes_W, imagenes_inversas):
    for i in range(len(imagenes)):
        cv2.imshow("Normal", imagenes[i]);
        cv2.imshow("Plano W", imagenes_W[i]);
        cv2.imshow("Inversas", imagenes_inversas[i]);
        cv2.waitKey(0);
    cv2.destroyAllWindows();  
    
# Función que verifica si los datos posee mapeo inverso.
def verificar_mapeo_inverso(a, b, c, d):
    op = (b * c) - (a * d);
    if(op == 0):
        return False
    else:
        return True
    

# Se aplica un umbral sobre la imagen para eliminar el ruido.
def umbral(imagenes_umbral, imagen):
    threshold_pixels = 127;
    max_Value = 255;
    i, nueva_imagen = cv2.threshold(imagen, threshold_pixels, max_Value, cv2.THRESH_BINARY);
    imagenes_umbral.append(imagen);
    name = "DR14-Threshold/threshold_image" + str(cont) + ".jpg";
    cv2.imwrite(name, nueva_imagen);
    imagenes_umbral.append(nueva_imagen);
    
# Se aplica un umbral adaptativo sobre la imagen para eliminar el ruido.
def umbral_adaptativo(imagenes_umbral, img_gray):
    background = 255;
    nueva_imagen = cv2.adaptiveThreshold(img_gray, background, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
    name = "DR14-Threshold-Adaptative/threshold_image" + str(cont) + ".jpg";
    cv2.imwrite(name, nueva_imagen);
    imagenes_umbral.append(nueva_imagen);
    
# Función encargada de hacer el mapeo bilineal del Plano Z al Plano W
def convertir_plano_w(imagenes_W, img, a, b, c, d):

    #Iniciación de parametros
    filas,columnas,canal = img.shape
    #Creación de imagen en blanco
    imagen_mapeada = np.zeros((filas,columnas,3),np.uint8)
    #Verificación de posibilidad de mapeo inverso
    if (es_inverso(a,b,c,d) == True):
        return imagen_mapeada
    #Creación de imagen en plano W a partir de Z
    for x in range(filas):
        for y in range(columnas):
            #Creación del Z
            pixel = complex(float(y), float(x))
            #Creación del W a partir de Z
            w = formula_W(pixel, a, b, c, d)
            #Verificación de Bordes
            if(((w.imag >= 0) and (w.imag < filas)) and ((w.real  >= 0) and (w.real < columnas))):
                for z in range(canal):
                    #Insersión de información a imagen nueva en R,G,B
                    imagen_mapeada.itemset((int(w.imag),int(w.real),z),img.item(x,y,z))
    name = "DR14-Preprocessed/proprecessed_image" + str(cont) + ".jpg";
    cv2.imwrite(name, imagen_mapeada);
    imagenes_W.append(imagen_mapeada);
        
# Formula Inversa
def formula_Inversa_W(w, a, b, c, d):
    op1 = (-1 * d) * w + b;
    op2 = c * w - a
    z = op1 / op2
    return z;

# Función del mapeo inverso de la imagen resultante.
def mapeo_inverso(imagenes_inversas, imagen_mapeo_inverso, imagen_original, a, b, c, d):
    filas, columnas, canal = imagen_mapeo_inverso.shape;
    imagen_inversa = np.zeros((filas, columnas, 3), np.uint8);
    
    for x in range(filas):
        for y in range(columnas):
            w = complex(y, x);
            z = formula_Inversa_W(w, a, b, c, d);
            for i in range(canal):
                if (imagen_mapeo_inverso[x, y, 0] == 0 and imagen_mapeo_inverso[x, y, 1] == 0
                    and imagen_mapeo_inverso[x, y, 2] == 0):
                    imagen_inversa.itemset((x, y, i), imagen_original.item(int(z.imag), int(z.real), i));
                else:
                    imagen_inversa.itemset((x, y, i), imagen_mapeo_inverso.item(x, y, i));
    name = "DR14-Inverse/inverse_image" + str(cont) + ".jpg";
    cv2.imwrite(name, imagen_inversa);
    imagenes_inversas.append(imagen_inversa);   
        
# Convertir todas las imagenes a plano w.
def convertir_W(imagenes_W, imagenes, a, b, c, d):
    global cont;
    cont = 0;
    for image in imagenes:
        height, width, channels = image.shape;
        height *= 2 ;
        width *= 2;
        cont += 1;
        convertir_plano_w(imagenes_W, image, a, b, c, d);

        
# Convertir todas las imagenes a plazo Z por medio del mapeo inverso.
def convertir_inversa(imagenes_inversas, imagenes_W, imagenes, a, b, c, d):
    global cont;
    cont = 0;
    for i in range(len(imagenes)):
        imagen_W = imagenes_W[i];
        imagen = imagenes[i];
        cont += 1;
        mapeo_inverso(imagenes_inversas, imagen_W, imagen, a, b, c, d);
        
        
# Convertir por el filtro de threshold
def filtro_umbral(imagenes_umbral, imagenes):
    global cont;
    cont = 0;
    for i in imagenes:
        umbral(imagenes_umbral, i);
        cont += 1
       
# Convertir por el filtro de threshold adaptativo.
def filtro_umbral_adaptativo(imagenes_umbral, imagenes_gris):
    global cont;
    cont = 0;
    for i in imagenes_gris:
        umbral_adaptativo(imagenes_umbral, i);
        cont += 1
    
######################################################################
# Aquí empieza el código.
######################################################################
    
a = complex(1.3, 0)
b = complex(0, 0)
c = complex(0, 0)
d = complex(1, 0) 

imagenes = [];
imagenes_gris = [];
imagenes_W = [];
imagenes_inversas = [];
imagenes_umbral = [];
imagenes_umbral_adaptativo = [];

obtener_imagenes(imagenes);
#ver_imagenes(imagenes);
convertir_W(imagenes_W, imagenes,  a, b, c, d);
#ver_imagenes(imagenes_W);
convertir_inversa(imagenes_inversas, imagenes_W, imagenes, a, b, c, d);
#ver_imagenes(imagenes_inversas);
filtro_umbral(imagenes_umbral, imagenes_inversas);
#ver_imagenes(imagenes, imagenes_W, imagenes_inversas)
obtener_gris(imagenes, imagenes_gris);

#filtro_umbral_adaptativo(imagenes_umbral, imagenes_gris);
#ver_imagenes()
print("Finalice");
