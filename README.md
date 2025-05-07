Aquí tienes una versión mejorada del `README.md`, con correcciones de estilo, ortografía, claridad y formato:

---

# Conversión de modelo YOLO (ONNX) a RKNN con RKNN Toolkit 2

Este repositorio permite convertir un modelo YOLO en formato ONNX al formato RKNN utilizando RKNN Toolkit 2, especialmente optimizado para funcionar en dispositivos con SoC RK3588S, como la Orange Pi 5 Pro.

## Conversión del modelo

Ejecuta el siguiente comando para convertir el modelo:

```bash
python -m rknn.api.rknn_convert -t rk3588s -i ./model_config.yml -o ./weights
```

---

## Instalación

### Prerrequisitos

* Dispositivo: Orange Pi 5 Pro
* Sistema Operativo: Instalar desde el repositorio [Joshua-Riek/ubuntu-rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip)

### Pasos de instalación

1. Actualiza los paquetes del sistema:

```bash
sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc
```

2. Crea y activa un entorno virtual:

```bash
python3 -m venv .env
source .env/bin/activate
```

3. Instala las dependencias de Python:

```bash
pip install -r requirements.txt
```

---

## Reconocimientos

* Código multihilo adaptado de: [leafqycc/rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
* Toolkit oficial de conversión RKNN: [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2/)
* SO usado en la orange pi 5 pro: [Joshua-Riek/ubuntu-rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip)

---

¿Quieres que también cree el archivo `requirements.txt` o algún ejemplo de `model_config.yml`?

