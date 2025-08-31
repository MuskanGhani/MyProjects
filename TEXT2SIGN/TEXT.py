import cv2
import time
import os

dataset_path = "C:/Users/HP/PyCharmMiscProject/TEXT2SIGN/asl_alphabet_test"

text = input("Enter text to convert into sign language: ").upper()

cv2.namedWindow("Sign Language", cv2.WINDOW_NORMAL)

for ch in text:
    if ch.isalpha():
        found = False
        # Loop through all files in the folder
        for filename in os.listdir(dataset_path):
            # If filename starts with the letter (case-insensitive)
            if filename.lower().startswith(ch.lower()):
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.imshow("Sign Language", img)
                    cv2.waitKey(1000)
                    found = True
                    break
        if not found:
            print(f"No image found for letter {ch}")
    else:
        print(f"Ignoring character: {ch}")

cv2.destroyAllWindows()
print("Done spelling word.")
