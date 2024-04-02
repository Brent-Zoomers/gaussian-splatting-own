import cv2
import os
import numpy as np

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2) + 0.000001
    max_pixel = 1.0
    psnr_image = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_image


def og_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_psnr_for_folders(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    psnr_values = []

    for file1, file2 in zip(files1, files2):
        img1 = cv2.imread(os.path.join(folder1, file1)) / 255.0
        img2 = cv2.imread(os.path.join(folder2, file2)) / 255.0
        img3 = cv2.imread(os.path.join(folder3, file2)) / 255.0


        
        if img1.shape != img2.shape:
            print(f"Images {file1} and {file2} have different dimensions. Skipping.")
            continue
        
        psnr_image = psnr(img1, img2)
        psnr2_image = psnr(img1, img3)

        psnr_value = og_psnr(img1, img2)


        print(psnr_value)


   
        psnr_image[np.isinf(psnr_image)] = 0
        gray = cv2.cvtColor((psnr_image/np.max(psnr_image)).astype(np.float32), cv2.COLOR_RGB2GRAY)
        original_height, original_width = gray.shape[:2]
        # te new dimensions by dividing by 2
        new_height = original_height // 2
        new_width = original_width // 2
        # Resize the image
        resized_image = cv2.resize(gray, (new_width, new_height))

        cv2.imshow(f'{file1}reg', resized_image)


        psnr2_image[np.isinf(psnr2_image)] = 0
        gray2 = cv2.cvtColor((psnr2_image/np.max(psnr2_image)).astype(np.float32), cv2.COLOR_RGB2GRAY)
        original_height2, original_width2 = gray2.shape[:2]
        # te new dimensions by dividing by 2
        new_height2 = original_height2 // 2
        new_width2 = original_width2 // 2
        # Resize the image
        resized_image2 = cv2.resize(gray2, (new_width2, new_height2))

        cv2.imshow(f'{file1}_baseline', resized_image2)

        # cv2.imshow(f'{file1}_diff', cv2.blur(resized_image2 - resized_image, (5, 5) ))
        cv2.imshow(f'{file1}_diff', resized_image2 - resized_image)



        key = cv2.waitKey()

        if key == ord('s'):  # 's' key pressed for save
            save_path = input("Enter filename to save (or 'q' to quit): ")
            if save_path != 'q':
                cv2.imwrite(f'{save_path}_0.001.png', resized_image*255)
                cv2.imwrite(f'{save_path}_baseline.png', resized_image2*255)
            print(f"Image saved as: {save_path}")
        cv2.destroyAllWindows()


        # psnr_values.append(psnr_value)
        # print(f"PSNR of {file1} and {file2}: {psnr_value}")

    return psnr_values

if __name__ == "__main__":
    folder1 = "eval/full_eval_3_0.001/counter/test/ours_30000/gt"
    folder2 = "eval/full_eval_3_0.001/counter/test/ours_30000/renders"
    folder3 = "eval/full_eval_3_0/counter/test/ours_30000/renders"

    psnr_values = calculate_psnr_for_folders(folder1, folder2)
    average_psnr = np.mean(psnr_values)
    print(f"Average PSNR across all images: {average_psnr}")
