"""
此專案是將out_of_focus deblur filter C版本的Code轉成Python來使用

參考資料：
OpenCV官方教學：
https://docs.opencv.org/4.x/de/d3c/tutorial_out_of_focus_deblur_filter.html
作者C版本
https://github.com/VladKarpushin/out_of_focus_deblur

作者測試的原始圖片(縮小的圖片會得到錯誤結果)
https://github.com/opencv/opencv/pull/12046#issuecomment-458472432

"""
import os
import cv2
import numpy as np

from pprint import pprint
import pdb


def show_img(win_name, img, w=1280, h=720):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    cv2.imshow(win_name, img)

def calcPSF(filterSizeHW, R):
    h= np.zeros(filterSizeHW, dtype= np.float32)
    center_point= (filterSizeHW[1] //2, filterSizeHW[0]//2)#center(x,y)
    cv2.circle(h, center_point, R, 255, -1, 8)
    
    summa= np.sum(h)        
    output_img= h / summa
    return output_img
    
def fftshift(PSF_img):
    output_img= PSF_img.copy()
    cx, cy= PSF_img.shape[1]//2 , PSF_img.shape[0]//2
    
    q0= PSF_img[0:cy, 0:cx].copy()
    q1= PSF_img[0:cy, cx:].copy()
    q2= PSF_img[cy:, 0:cx].copy()
    q3= PSF_img[cy:, cx:].copy()

    output_img[0:cy, 0:cx]= q3[:]#(q3 copy to q0_region)
    output_img[cy:, cx:]= q0[:]#(q0 copy to q3_region)
    output_img[0:cy, cx:]= q2[:]#(q2 copy to q1_region)
    output_img[cy:, 0:cx]= q1[:]#(q1 copy to q2_region)

    return output_img




def calcWienerFilter(PSF_img, snr):
    PSF_shift_img= fftshift(PSF_img)
    planes= cv2.merge([PSF_shift_img.copy(), np.zeros_like(PSF_shift_img)])#merge to -->(H,W,2)
    
    complexl= cv2.dft(planes)
    planes= cv2.split(complexl)
    planes_0= planes[0]
    
    denom= cv2.pow(np.abs(planes_0), 2)
    denom += snr
    output= cv2.divide(planes_0, denom)
    return output
    
    
def filter2DFreq(img, Hw_wiener_filter):
    planes= cv2.merge([img.astype(np.float32), np.zeros_like(img, dtype=np.float32)])#merge to -->(H,W,2)
    # pdb.set_trace()
    complexl= cv2.dft(planes, flags= cv2.DFT_SCALE)

    complexH= cv2.merge([Hw_wiener_filter, np.zeros_like(Hw_wiener_filter)])
    complexlH= cv2.mulSpectrums(complexl, complexH, 0)

    complexlH= cv2.idft(complexlH)
    output_img= cv2.split(complexlH)[0]
    return output_img







def main():
    img_path= "original.jpg"
    img_ori= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    ##Wiener Filter Parameters
    R= 53 #circular PSF parameters
    snr= 5200 #signal-to-noise ratio


    while True:
        img= img_ori.copy()
        #It needs to process even image only    
        filterSizeHW= (img.shape[0] & -2), (img.shape[1] & -2)#強制變成偶數

        #Hw calculation
        PSF_img= calcPSF(filterSizeHW= filterSizeHW, R= R)
        
        Hw_wiener_filter= calcWienerFilter(PSF_img, snr= 1.0 / snr)
        
        #Filtering
        imgOut= filter2DFreq(img[:filterSizeHW[0], :filterSizeHW[1]], Hw_wiener_filter)
        imgOut= cv2.convertScaleAbs(imgOut)# imgOut= imgOut.astype(np.uint8)-->會有破圖
        cv2.normalize(imgOut.copy(), imgOut, 0, 255, cv2.NORM_MINMAX)
        
        

        show_img("Frame", img)        
        show_img("ImgOut", imgOut)
        key= cv2.waitKey()

        if key==ord('r'):
            R+=1
        elif key==ord('R'):
            R-=1
        elif key==ord('s'):
            snr+=10
        elif key==ord('S'):
            snr-=10
        elif key==27:#Esc
            break
        print("R: {}, snr: {}".format(R, snr))

    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
