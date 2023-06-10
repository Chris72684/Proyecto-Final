import cv2
import os
import imutils
import Funciones

Nombre = input('Nombre de usuario: ')
Direccion = 'C:/Users/USUARIO/Documents/Proyecto Vision Artificial/Usuarios'
DirecUsuario = Direccion + '/' + Nombre
#print('la direccion es:' + DirecUsuario)
if not os.path.exists(DirecUsuario):
    print('Carpeta Creada: ', DirecUsuario)
    os.makedirs(DirecUsuario)
    Peticion, direct = Funciones.captura(DirecUsuario)
else:
    print('Usuario existente')
    YN = input('¿Añadir video? (Y/N): ')
    if(YN == 'Y' or YN == 'y'):
        Peticion, direct = Funciones.captura(DirecUsuario)
    else:
        print('Saliendo...')
        Peticion=False
        #Añadir opciones de salida...
if Peticion==True:
   DirecImg = Funciones.imagen(direct, DirecUsuario)
print('comenzando entrenamiento')
Funciones.entrenamiento(DirecImg, DirecUsuario, Nombre)
