# ¿Cómo obtener el modelo Darknet para ejecutarlo en RK3588S con RKNN Toolkit 2? (versión fácil)

## Requisitos previos

* Python 3.10
* `venv` para entornos virtuales

## Pasos

### 1. Clonar el repositorio de YOLOv7

```bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
```

### 2. Crear y activar un entorno virtual

```bash
python3.10 -m venv .env
source .env/bin/activate
```

### 3. Descargar los pesos preentrenados de YOLOv7-tiny

Dentro de la carpeta `yolov7`, ejecuta:

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

### 4. Exportar a ONNX

Ejecuta el siguiente comando para convertir el modelo `.pt` a `.onnx`:

```bash
python export.py --weights ./yolov7-tiny.pt --simplify --img-size 640 640
```

> ⚠️ **Importante**: No utilices opciones como `--nms` ni ninguna otra opción adicional del script `export.py`. El `RKNN Toolkit 2` **no es compatible** con esas funciones.

### 5. Exportar a formato `.rknn`

Con el modelo `.onnx` generado, puedes seguir las instrucciones del README principal para convertirlo a formato `.rknn` utilizando `RKNN Toolkit 2`.

---

## Referencias

* [yolov7](https://github.com/WongKinYiu/yolov7)
* [YOLOv7 ONNX en Google Colab](https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb#scrollTo=eWlHa1NJ-_Jw)

