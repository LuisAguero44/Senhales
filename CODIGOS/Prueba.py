import cv2
import numpy as np
import tensorflow as tf
import SeguimientoManos as sm  


model = tf.keras.models.load_model("modelo_gestos.h5")


detector = sm.detectormanos(Confdeteccion=0.8)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

class_names = list(model.class_names) 

while True:
   
    ret, frame = cap.read()
    
    frame = detector.encontrarmanos(frame)
    lista, bbox = detector.encontrarposicion(frame, dibujarPuntos=False, dibujarBox=True)
    
    
    if bbox:
        
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin = max(0, xmin - 20), max(0, ymin - 20)  
        xmax, ymax = min(frame.shape[1], xmax + 20), min(frame.shape[0], ymax + 20)
        
        recorte = frame[ymin:ymax, xmin:xmax]
        recorte = cv2.resize(recorte, (224, 224))  
        recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB) 
        recorte = recorte / 255.0 
        
        recorte = np.expand_dims(recorte, axis=0) 
        prediccion = model.predict(recorte)
        clase_index = np.argmax(prediccion)
        clase_gesto = class_names[clase_index]
        
        
        cv2.putText(frame, clase_gesto, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    
    cv2.imshow("Reconocimiento de Gestos", frame)
    
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
