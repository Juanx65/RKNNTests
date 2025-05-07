#!/bin/bash

echo "Mostrando uso actual de la NPU. Presiona Ctrl+C para salir."
while true; do
    echo -n "Uso actual de la NPU: "
    sudo cat /sys/kernel/debug/rknpu/load
    sleep 1
done

