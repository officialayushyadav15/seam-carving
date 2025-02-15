# Image-Stitching-using-Seam-Carving
# 🌄 Seam Carving Panorama Stitching

### 🖼️ A Python program that stitches two images into a seamless panorama using seam carving.

## 📌 Overview  
This project implements **panorama stitching** using **seam carving**, an energy-based technique that finds the optimal overlap between images and removes unnecessary seams for smooth blending.

## ⚡ Features  
✔ **Loads & Displays Images** – Reads and visualizes input images using `OpenCV` and `Matplotlib`  
✔ **Resizes Images to Same Height** – Ensures proper alignment before stitching  
✔ **Finds Overlapping Region** – Determines the best overlap area for blending  
✔ **Computes Energy Map** – Uses `Sobel` filters to detect seam importance  
✔ **Finds & Removes Optimal Seam** – Applies dynamic programming to carve the best seam  
✔ **Blends Overlapping Region with Multi-Band Blending** – Uses a frequency-based approach for high-quality blending, ensuring seamless transitions  
✔ **Saves the Final Output** – Stores the stitched image as `panorama_seam_carving.png`  


## Install Dependencies  
Ensure you have Python installed, then run:  

```sh
pip install opencv-python numpy matplotlib
```

## 🛠️ Requirements  
- Python 3.x  
- OpenCV (`opencv-python`)  
- NumPy  
- Matplotlib

## 📸 Sample Results  

### 🎯 Input Images  
| Image 1  |  Image 2  |
|:--------:|:--------:|
| ![Input 1](Samples/1.jpg) | ![Input 2](Samples/2.jpg) |

### 🏆 Output Panorama  
![Stitched Panorama](Samples/panorama_seam_carving.png)



