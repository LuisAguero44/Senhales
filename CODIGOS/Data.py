import cv2
import os
import SeguimientoManos as sm

nombre='Gesto_1'
direccion='C:\Users\User\Desktop\LenguajeSeñas\Data'

carpeta=direccion+ '/'+nombre

if not os.path.exists(carpeta):
  print('Carpeta creada')
  os.makedirs(carpeta)

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

cont=0
detector=sm.detectormanos(Confdeteccion=0.9)

while True:
  ret,frame=cap.read()
  frame=detector.encontrarmanos(frame, dibujar=True)
  lista,bbox,mano=detector.encontrarposicion(frame, ManoNum=0,dibujarPuntos=False,dibujarBox=True,color=[0,255,0])
  if mano==1:
   xmin,ymin,xmax,ymax=bbox
   xmin=xmin-40
   ymin=ymin-40
   xmax=xmax+40
   ymax=ymax+40

   recorte = frame[ymin:ymax,xmin:xmax]
   recorte = cv2.resize(recorte,(224,224))
   cv2.imshow("recorte",recorte)
   cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)

    cv2.imwrite(carpeta+'/'+str(count)+'.jpg',recorte)
    count=count+1
  
  cv2.imshow("Señas",frame)
  t=cv2.waitKey(1)
  if t==27 or cont==150:
    break

cap.release()
cv2.destroyAllWindows()


