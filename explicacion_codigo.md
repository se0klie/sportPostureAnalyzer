# Explicación de `main.py`

Este script analiza un video de ejercicio con MediaPipe Pose Landmarker y calcula ángulos articulares por fotograma para luego devolver una estructura con esos valores.

## Qué hace, en resumen

1. Abre el modelo `pose_landmarker_full.task`.
2. Lee el video `./videos/squat.mp4`.
3. Detecta el esqueleto de la persona en cada frame.
4. Guarda las coordenadas de los 33 puntos corporales que entrega MediaPipe.
5. Calcula ángulos para articulaciones concretas según el ejercicio elegido.
6. Imprime un diccionario con los ángulos por marca de tiempo.

## Estructura general

El archivo tiene dos bloques principales:

- Extracción de landmarks desde el video.
- Cálculo de ángulos a partir de esos landmarks.

Al final, el script ejecuta ambas partes en secuencia:

- primero llama a `extract_pose_vectors(video_squat)`
- luego llama a `angle_calculation(landmarks_history, 'squat')`
- y finalmente hace `print(angles)`

## Dependencias y recursos

El script usa estas librerías:

- `numpy`: para operaciones vectoriales y trigonométricas.
- `mediapipe`: para detectar puntos del cuerpo.
- `cv2`: para abrir el video y convertir cada frame a RGB.
- `time`: está importado, pero en el código actual no se usa.

También depende de dos archivos locales:

- `pose_landmarker_full.task`: modelo de MediaPipe.
- `videos/squat.mp4`: video de entrada.

## Configuración de ejercicios

La constante `EXERCISE_CONFIG` define qué articulaciones se analizan para cada ejercicio y qué tres puntos corporales forman el ángulo.

Ejemplos:

- `squat` usa `knee` y `hip`.
- `push-up` usa `elbow` y `hip`.
- `plank` usa `hip` y `shoulder`.

Cada entrada guarda una lista de tres índices de landmarks:

- primer punto: extremo superior o inicial del vector
- segundo punto: vértice del ángulo
- tercer punto: extremo inferior o final del vector

Por ejemplo, para `knee: [24, 26, 28]` el ángulo se calcula con:

- 24: cadera
- 26: rodilla
- 28: tobillo

## Cómo extrae la pose del video

La función `extract_pose_vectors(video)` recorre el video frame por frame.

### Flujo interno

1. Crea una lista vacía llamada `video_landmarks_history`.
2. Abre un contexto con `PoseLandmarker.create_from_options(pose_opt)`.
3. Mientras el video siga abierto:
   - lee un frame
   - lo invierte horizontalmente con `cv2.flip(frame, 1)`
   - lo convierte a RGB
   - lo empaqueta como `mp.Image`
   - obtiene el timestamp en milisegundos
   - ejecuta `detect_for_video(...)`
4. Si MediaPipe detecta landmarks:
   - toma la primera persona detectada
   - recorre los 33 landmarks
   - guarda para cada punto su `id`, `x`, `y`, `z` y `visibility`
5. Agrega un registro por frame con:
   - `timestamp_ms`
   - `landmarks`
6. Libera el video y devuelve todo el historial.

### Estructura de salida

La salida de esa función es una lista de diccionarios como esta:

```python
[
    {
        'timestamp_ms': 123,
        'landmarks': [
            {'id': 0, 'x': ..., 'y': ..., 'z': ..., 'visibility': ...},
            ...
        ]
    }
]
```

## Cómo calcula los ángulos

La función `angle_calculation(history, exercise)` toma el historial anterior y un ejercicio como `squat`.

### Flujo interno

1. Busca la configuración del ejercicio en `EXERCISE_CONFIG`.
2. Recorre cada frame del historial.
3. Si el frame no tiene landmarks o no llega a 33 puntos, lo salta.
4. Para cada articulación definida en el ejercicio:
   - toma los tres índices del vector
   - extrae sus coordenadas `x` e `y`
   - construye dos vectores desde el punto central hacia los otros dos puntos
   - calcula el ángulo entre esos dos vectores con `arctan2`
5. Ajusta el resultado para que quede en grados entre 0 y 180.
6. Guarda los ángulos de ese frame en un diccionario indexado por timestamp.

### Idea matemática

El código calcula el ángulo entre dos segmentos del cuerpo usando trigonometría 2D.

Primero forma dos vectores:

- `vector_u = upper - apex`
- `vector_v = lower - apex`

Luego obtiene sus ángulos absolutos con `arctan2`, calcula la diferencia y la transforma a grados.

En términos simples, mide cuánto se abre o cierra una articulación en cada frame.

## Qué significa la salida final

La variable `angles` termina siendo un diccionario con esta forma:

```python
{
    timestamp_ms_1: {'knee': 92.4, 'hip': 143.1},
    timestamp_ms_2: {'knee': 88.7, 'hip': 139.8},
}
```

Cada clave es una marca de tiempo del video y cada valor contiene los ángulos calculados para las articulaciones definidas en ese ejercicio.

## Qué intenta analizar en un squat

En el caso actual, el script termina usando `squat`, así que calcula principalmente:

- el ángulo de la rodilla
- el ángulo de la cadera

Eso sirve para estimar la profundidad y la alineación general del movimiento.

## Observaciones importantes

- El script no dibuja nada en pantalla; solo procesa el video y crea datos numéricos.
- `time` está importado pero no se usa.
- `video_squat` queda fijado a `./videos/squat.mp4`, así que el script no está parametrizado para otros videos todavía.
- Aunque existe configuración para `push-up` y `plank`, el flujo actual solo ejecuta `squat`.
- El cálculo usa solo coordenadas `x` e `y`; el valor `z` se guarda, pero no participa en el ángulo.

## En una frase

Este programa detecta la postura de una persona en un video, extrae los puntos corporales relevantes y calcula ángulos articulares para evaluar el movimiento, en este caso un squat.