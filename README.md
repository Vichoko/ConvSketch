# ConvSketch
 Clasificación y búsqueda por similitud de sketches usando redes convolucionales

## Enunciado

Leer ./enunciado.pdf

## Informe

Leer ./Informe.pdf

## Dependencias

### Datos

Para funcionar se requieren los datos en formato TFRecord, que se descargan de aqui:
https://users.dcc.uchile.cl/~voyanede/cc6204/quickdraw/data/

Luego, hay que colocar los 3 archivos en algun directorio.
Los archivos necesarios son:
- mean.dat
- test.tfrecords
- train.tfrecords

Además hay que configurar la ruta para el correcto funcionamiento.

### Modelos entrenados

Para evaluar clasificador y/o busqueda por similitud,
 se requiere además de los TFRecord, descargar los modelos entrenados.

Existen dos modelos disponibles:

#### skNet
https://users.dcc.uchile.cl/~voyanede/cc6204/quickdraw/models/sknet/

#### skResNet
https://users.dcc.uchile.cl/~voyanede/cc6204/quickdraw/models/skresnet/

Sea cual sea el modelo, se necesitan los 4 archivos en algun directorio:
- checkpoint
- model.ckpt-10000.data[...]
- model.ckpt-10000.index
- model.ckpt-10000.meta

Además hay que configurar la ruta y el net_type para el correcto
funcionamiento.

## Uso

### Entrenamiento clasificador

1. Descargar TFRecords y Mean, y colocar en algun directorio
2. Configurar ruta ```data_dir``` en ```configuration_sketch.py```
3. Elegir arquitectura ```net_type``` en ```configuration_sketch.py```
4. Ejecutar ```python train_sketch_net.py -mode train -device gpu```
5. Leer progreso en stdout

### Evaluacion clasificador

1. Descargar modelo entrenado (skNet o skResNet) y colocar archivos
en algun directorio
2. Configurar ruta ```snapshot_prefix``` a directorio que contenga modelo
3. Elegir arquitectura ```net_type``` en ```configuration_sketch.py```
3. Ejecutar ```python train_sketch_net.py -mode evaluate -device gpu```
5. Leer metricas en stdout

### Evaluacion Busqueda por Similitud (mAP)

1. Descargar modelo entrenado (skNet o skResNet) y colocar archivos
en algun directorio
2. Configurar ruta ```snapshot_prefix``` a directorio que contenga modelo
3. Elegir arquitectura ```net_type``` en ```configuration_sketch.py```
3. Ejecutar ```python train_sketch_net.py -mode train -device gpu```
5. Leer metricas en stdout
