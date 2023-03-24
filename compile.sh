#!/bin/bash
nvcc -G -o program src/*.cpp src/*.cu -Iinc
