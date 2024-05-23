import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Para usar o vídeo
#cap = cv2.VideoCapture('hall_box_battery_mp2.mp4')

# As 3 próximas linhas são para usar a webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def identifica_cor(frame, mode):
    '''
    Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
    '''

    # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao 
    # vermelho puro (H=0) estão entre H=-8 e H=8. 
    # Veja se este intervalo de cores está bom
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == 0:
        cor_menor = np.array([0, 50, 50])
        cor_maior = np.array([5, 255, 255])

    if mode == 1:
        cor_menor = np.array([6, 50, 50])
        cor_maior = np.array([10, 255, 255])

    segmentado_cor = cv2.inRange(frame_hsv, cor_menor, cor_maior)

    # Será possível limpar a imagem segmentado_cor? 
    # Pesquise: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5,5),np.uint8)
    segmentado_cor = cv2.morphologyEx(segmentado_cor, cv2.MORPH_OPEN, kernel)


    # Encontramos os contornos na máscara e selecionamos o de maior área
    # Encontramos os contornos na máscara e selecionamos o de maior área
    contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_contorno = None
    maior_contorno_area = 0

    cv2.drawContours(frame, contornos, -1, [255, 0, 255], 5)


    #########
    if mode == 1:
        #print(img[200][200])
        cv2.putText(img, str(frame_hsv[200][200]), (400,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        cv2.rectangle(img, (200, 200), (201,201), (0, 0, 255))

    for cnt in contornos:
        area = cv2.contourArea(cnt)


        if area > maior_contorno_area:
            maior_contorno = cnt
            maior_contorno_area = area

    
    # Encontramos o centro do contorno fazendo a média de todos seus pontos.
    if not maior_contorno is None :

        x,y,w,h = cv2.boundingRect(maior_contorno)
        
        

        if h > (w*1.5):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            
            distancia = h*h*0.001543 - 1.313*h + 324.53

            cv2.putText(img, "Distancia: " + str(int(distancia)) + "cm", (20,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img, "Posicione a caixa na vertical!", (20,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

        cv2.drawContours(frame, [maior_contorno], -1, [0, 0, 255], 5)
        maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
        media = maior_contorno.mean(axis=0)
        media = media.astype(np.int32)
        cv2.circle(frame, tuple(media), 5, [0, 255, 0])
    else:
        media = (0, 0)

    cv2.imshow('', frame)
    cv2.imshow('imagem in_range', segmentado_cor)
    cv2.waitKey(1)

    centro = (frame.shape[0]//2, frame.shape[1]//2)

    

    return media, centro

mode = 0

while(True):
    # Capture frame-by-frame
    #print("Novo frame")
    ret, frame = cap.read()
    

    img = frame.copy()

    media, centro = identifica_cor(img, mode)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    

    # Display the resulting frame
    cv2.imshow('original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('p'):
        mode += 1
        mode = mode%2
        print("MUDOU O MODO!!!!!!!!!!!!!!!")


    #print("No circles were found")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()