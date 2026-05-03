#!/bin/bash

blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"

include_count=$(find include  -type f \( -name "*.cpp" -o -name "*.h" \) -exec cat {} \; | wc -l)
src_count=$(find src  -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.txt" \) -exec cat {} \; | wc -l)
total=$((include_count + src_count))

if [ ! -d "data/debug" ]; then
    mkdir -p data/debug
    touch data/debug/here_save_debug_images
fi

if [ ! -d "data/video" ]; then
    mkdir -p data/video
    touch data/video/here_save_video
fi

if [ ! -d "data/speed" ]; then
    mkdir -p data/speed
    touch data/speed/here_save_shoot_speed
fi

if [ ! -d "/etc/openrm" ]; then 
    mkdir -p /etc/openrm
    sudo cp -r data/uniconfig/* /etc/openrm/
    sudo chmod -R 777 /etc/openrm
fi

if [ ! -d "config" ]; then 
    ln -s /etc/openrm ./config
fi


if [ ! -d "build" ]; then 
    mkdir build
fi

imshow=0
usensys=0
skip_build=0

while getopts ":rcg:lsnk" opt; do
    case $opt in
        r)
            echo -e "${yellow}<--- delete 'build' --->\n${reset}"
            sudo rm -rf build
            mkdir build
            shift
            ;;

        c)
            sudo cp -r data/uniconfig/* /etc/openrm/
            sudo chmod -R 777 /etc/openrm
            exit 0
            shift
            ;;
        g)
            git_message=$OPTARG
            echo -e "${yellow}\n<--- Git $git_message --->${reset}"
            git pull
            git add -A
            git commit -m "$git_message"
            git push
            exit 0
            shift
            ;;

        l)
            cd ../OpenRM
            sudo ./run.sh
            cd ../TJURM-2024
            exit 0
            shift
            ;;
        s)
            imshow=1
            shift
            ;;
        n)
            usensys=1
            shift
            ;;
        k)
            skip_build=1
            shift
            ;;
        \?)
            echo -e "${red}\n--- Unavailable param: -$OPTARG ---\n${reset}"
            ;;
        :)
            echo -e "${red}\n--- param -$OPTARG need a value ---\n${reset}"
            ;;
        esac
    done


cd build # 无论是否编译，都需要进入 build 目录以供后续复制可执行文件

if [ $skip_build -eq 0 ]; then
    echo -e "${yellow}<--- Start CMake --->${reset}"
    cmake ..

    echo -e "${yellow}\n<--- Start Make --->${reset}"
    max_threads=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    make -j "$max_threads"
else
    echo -e "${yellow}\n<--- Skip CMake & Make (Using existing binaries) --->${reset}"
fi


echo -e "${yellow}\n<--- Total Lines --->${reset}"
echo -e "${blue}        $total${reset}"

function restore_performance() {
    echo -e "${yellow}\n<--- Exiting & Restoring System Settings --->${reset}"
    sudo jetson_clocks --restore
    sudo systemctl restart nvfancontrol
    echo -e "${blue}System performance restored to normal.${reset}"
}

trap restore_performance EXIT

echo -e "${yellow}\n<--- Enabling Max Performance Mode --->${reset}"
sudo nvpmodel -m 8

if ! sudo test -f /root/.jetsonclocks_conf.txt; then
    echo -e "${blue}Storing default jetson_clocks state...${reset}"
    sudo jetson_clocks --store
fi

sudo jetson_clocks --fan

echo -e "${yellow}\n<--- Run Code --->${reset}"
sudo rm -f /usr/local/bin/TJURM-2024
sudo cp TJURM-2024 /usr/local/bin/
sudo pkill TJURM-2024
sudo chmod 777 /dev/tty*

if [ $usensys = 1 ]; then
    echo -e "${yellow}\n<--- Start Nsight Systems Profiling --->${reset}"
    if [ $imshow = 1 ]; then
        sudo nsys profile \
            --trace=cuda,nvtx,osrt \
            -y 10 -d 10 \
            --output=autoaim \
            --force-overwrite=true \
            TJURM-2024 -s
    else
        sudo nsys profile \
            --trace=cuda,nvtx,osrt \
            -y 10 -d 10 \
            --output=autoaim \
            --force-overwrite=true \
            TJURM-2024
    fi
else
    echo -e "${yellow}\n<--- Start Program (Normal Mode) --->${reset}"
    if [ $imshow = 1 ]; then
        sudo TJURM-2024 -s
    else
        sudo TJURM-2024
    fi
fi

/etc/openrm/guard.sh

echo -e "${yellow}<----- OVER ----->${reset}"