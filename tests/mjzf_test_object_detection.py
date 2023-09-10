import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# Definimos un pipeline
pipeline = depthai.Pipeline()

# Agregamos una ColorCamera
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(672,384) #Usaremos el preview output con una resolución de 300 por 300
cam_rgb.setInterleaved(False)

# Agregamos un nodo para una red de detección
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# Definimos el path para el BLOB (modelo de redes neuronales) detection_nn.setBlobPath("/path/to/model.blob")
# Usaremos el blobconverter para convertir y descargar el modelo blobconverter.from_zoo()
#detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setBlobPath(blobconverter.from_zoo(name='vehicle-detection-adas-0002', shaves=6))
#Debemos definir el threshold para filtrar adecuadamente los resultados
detection_nn.setConfidenceThreshold(0.5)

#Conectamos el preview de nuestra colorCamera a la red neuronal
cam_rgb.preview.link(detection_nn.input)

#Ahora, queremos recibir la imagen de la camara y a inferencia del modelo
#así que usaremos xLinkOut porque queremos pasar la información del dispositivo a al host
# Imagen
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)
#Inferencia del modelo
xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

#Ahora podemos iniciar nuestro pipeline
with depthai.Device(pipeline) as device:
    
    #Definiremos una cola del output del lado del host para acceder a los datos
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    #Ahora consumiremos los resultados accedidos
    #Definimos placeholders
    frame = None
    detections = []

    #Necesitamos una función para normalizar el output de la red neuronal
    #ya que recibimos valores entre 0 y 1 y lo necesitamos en pixeles
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    #Ahora estamos listos para ejecutar nuestro programa
    while True:

        #Obtenemos los ultimos resultados del nodo de la red neuronal y de la camara
        in_rgb = q_rgb.tryGet() #Usamos tryGet para que nos devuelva el ultimo resultado o None si la cola está vacía
        in_nn = q_nn.tryGet()

        #Obtenermos el rgb de la camara 
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        #Obtenemos los resultados de la red neuronal
        if in_nn is not None:
            detections = in_nn.detections #Obtenemos las detecciones

        #Presentamos los resultados
        if frame is not None:
            for detection in detections:
                #Normalizamos las ubicaciones del cuadro
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                #Escribimos el cuadro
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)
        
        #Si usamos la 'q', cerramos el programa
        if cv2.waitKey(1) == ord('q'):
            break
