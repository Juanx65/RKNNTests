# Para transformar modelo yolo ( en formato onnx )  a rknn toolkit 2

```
python -m rknn.api.rknn_convert -t rk3588s -i ./model_config.yml -o ./weights
```

# Instalation

## prerequisitos:

Orange Pi 5 Pro

Instalar SO de este [github repository](https://github.com/Joshua-Riek/ubuntu-rockchip) especifico para la Orange Pi 5 Pro

correr

sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc

venv

python -m venv .env
source .env/bin/activate

pip install -r requirements


