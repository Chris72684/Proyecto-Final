#Pruebas
import time
import cv2
import os
import imutils
import numpy as np
def captura(direc):
    direct = direc+'/Video'
    if not os.path.exists(direct):
        os.makedirs(direct)
    cam = cv2.VideoCapture(0)
    if (cam.isOpened() == False):
        print('La camara no se pudo usar')
        return (False, direct)
    else:
        print('video iniciado')
    ancho = int(cam.get(3))
    alto = int(cam.get(4))
    video = cv2.VideoWriter_fourcc(*'mp4v')
    salida = cv2.VideoWriter(direct + '/video.mp4',
                             video,
                             30,
                             (ancho, alto))
    
    inicio = time.time()
    fin = time.time()
    des = 0

    while(fin-inicio<10):
        fin = time.time()
        ret, frame = cam.read()
        if ret == True:
            salida.write(frame)
        cont = int(10-(fin-inicio))
        if des!=cont:
            des = cont
            print(cont)

    cam.release()
    salida.release()
    cv2.destroyAllWindows()
    print('video guardado')
    '''if fun!= False:
        imagen()'''
    return (True, direct+'/video.mp4')
def imagen(direct, DirecUsuario):
    DirecImg = DirecUsuario + '/Imagenes'
    if not os.path.exists(DirecImg):
        os.makedirs(DirecImg)
    video = cv2.VideoCapture(direct)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = video.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for(x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (59, 36, 209), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro,(150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(DirecImg+'/imagen_{}.jpg'.format(count), rostro)
            count = count + 1
        #cv2.imshow('frame', frame)
    if count<150:
        print('video insuficiente\ Tomando nuevo video')
        Tr, dir =captura(DirecUsuario)
        imagen(dir, DirecUsuario)
    print('Imagenes Suficientes')
    video.release()
    #cv2.destroyAllWindows()
    return(DirecImg)
def entrenamiento(DirecImg, DirecUsuario, Usuario):
    facesData = []
    labels = []
    for fileName in os.listdir(DirecImg):
        print('Reading: ', fileName)
        labels.append(0)
        facesData.append(cv2.imread(DirecImg + '/' + fileName, 0))
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    print('Entrenando...')
    face_recognizer.train(facesData, np.array(labels))

    face_recognizer.write(DirecUsuario+'/modelo_'+Usuario+'.xml')
    print('Modelo almacenado')
