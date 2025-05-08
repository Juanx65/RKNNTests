# ⚠️ Por favor, cambia al usuario root

# 📌 Fijar frecuencia de la CPU

echo "Frecuencias disponibles para CPU0-3:"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies

sudo echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
sudo echo 1800000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed

echo "Frecuencia actual de CPU0-3:"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq


echo "Frecuencias disponibles para CPU4-5:"
sudo cat /sys/devices/system/cpu/cpufreq/policy4/scaling_available_frequencies

sudo echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
sudo echo 2400000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed

echo "Frecuencia actual de CPU4-5:"
sudo cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq


echo "Frecuencias disponibles para CPU6-7:"
sudo cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies

sudo echo userspace > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
sudo echo 2400000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed

echo "Frecuencia actual de CPU6-7:"
sudo cat /sys/devices/system/cpu/cpufreq/policy6/cpuinfo_cur_freq


# ⚙️ Fijar frecuencia de la NPU

echo "Frecuencias disponibles para la NPU:"
sudo cat /sys/class/devfreq/fdab0000.npu/available_frequencies    

sudo echo userspace > /sys/class/devfreq/fdab0000.npu/governor
sudo echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq

echo "Frecuencia actual de la NPU:"
sudo cat /sys/class/devfreq/fdab0000.npu/cur_freq

