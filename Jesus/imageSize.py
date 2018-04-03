import os
import cv2 

# Go the the train images directory and load them PISTOLS!
os.chdir("C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train\\Pistol")
file_names = os.listdir()
cnt = 0 
# print(file_names)
pistol_max_width = 0
pistol_min_width = 9999
pistol_max_height = 0
pistol_min_height = 9999

for im_name in file_names:
    cnt = cnt + 1
    file_name = os.path.splitext(im_name)
    file_extension = file_name[1]
    
    if file_extension == ".jpg":
        n_image = cv2.imread(im_name)
        shape = n_image.shape

        # print(shape[0])

        if shape[0] > pistol_max_width:
            pistol_max_width = shape[0]

        elif shape[0] < pistol_min_width:
            pistol_min_width = shape[0]
        
        if shape[1] > pistol_max_height:
            pistol_max_height = shape[1]
        
        elif shape[1] < pistol_min_height:
            pistol_min_height = shape[1]

print("Pistol max. heigth: ",pistol_max_height,"x",pistol_max_width,"px")
print("Pistol min. heigth: ",pistol_min_height,"x",pistol_min_width,"px")

# Go the the train images directory and load them PISTOLS!
os.chdir("C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train\\Smartphone")
file_names = os.listdir()

# print(file_names)
phone_max_width = 0
phone_min_width = 9999
phone_max_height = 0
phone_min_height = 9999

for im_name in file_names:
    cnt = cnt + 1
    file_name = os.path.splitext(im_name)
    file_extension = file_name[1]
    
    if file_extension == ".jpg":
        n_image = cv2.imread(im_name)
        shape = n_image.shape

        # print(shape[0])

        if shape[0] > phone_max_width:
            phone_max_width = shape[0]

        elif shape[0] < phone_min_width:
            phone_min_width = shape[0]
        
        if shape[1] > phone_max_height:
            phone_max_height = shape[1]
        
        elif shape[1] < phone_min_height:
            phone_min_height = shape[1]

print("Smartphone max: ",phone_max_height,"x",phone_max_width,"px")
print("Smartphone min: ",phone_min_height,"x",phone_min_width,"px")
print("Number of train images: ",cnt)