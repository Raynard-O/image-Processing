#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "main.h"
#include "NumCpp.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

auto normalise(auto img, auto mean, char std)
{
    auto normed = (img - nc::mean(img)) / (nc::std(img));
    return normed;
}

auto ridge_segment(auto im, auto blksze, char thresh)
{

    auto cols = im.shape;
    auto rows = im.shape;

    auto im = normalise(im, 0, 1); //normalise to get zero mean and unit standard deviation

    auto new_rows = nc::int(blksze * nc::ceil((nc::float(rows)) / (nc::float(blksze))));
    auto new_cols = nc::int(blksze * nc::ceil((nc::float(cols)) / (nc::float(blksze))));

    auto padded_img = nc::zeros((new_rows, new_cols));
    auto stddevim = nc::zeros((new_rows, new_cols));

    im = padded_img [0:rows][:, 0:cols];

    for (int i = 0, new_rows, blksze)
    {
        for (int j = 0, new_cols, blksze)
        {
            block = padded_img [i:i + blksze][:, j:j + blksze];
            stddevim [i:i + blksze][:, j:j + blksze] = nc::std(block) * nc::ones(block.shape);
        }
    }
    stddevim = stddevim [0:rows][:, 0:cols];

    auto mask = stddevim > thresh;

    auto mean_val = nc::mean(im[mask]);

    auto std_val = nc::std(im[mask]);

    auto normim = (im - mean_val) / (std_val);

    return (normim, mask);
}

auto ridge_orient(auto im, auto gradientsigma, auto blocksigma, auto orientsmoothsigma)
{
    auto cols = im.shape;
    auto rows = im.shape;

    //Calculate image gradients.
    auto sze = nc::fix(6 * gradientsigma);
    if
        nc::remainder(sze, 2) == 0 : sze = sze + 1;
    auto gauss = cv2.getGaussianKernel(nc::int(sze), gradientsigma);
    f = gauss * gauss.T;

    int fy, fx = nc::gradient(f);
    //#Gradient of Gaussian int Gx = signal.convolve2d(im, fx, mode = 'same');
    int Gy = signal.convolve2d(im, fy, mode = 'same');

    int Gxx = nc::power(Gx, 2);
    int Gyy = nc::power(Gy, 2);
    int Gxy = Gx * Gy;

    //Now smooth the covariance data to perform a weighted summation of the data.

    sze = nc::fix(6 * blocksigma);
    gauss = cv2.getGaussianKernel(nc::int(sze), blocksigma);
    f = gauss * gauss.T;

    Gxx = ndimage.convolve(Gxx, f);
    Gyy = ndimage.convolve(Gyy, f);
    Gxy = 2 * ndimage.convolve(Gxy, f);
    // Analytic solution of principal direction
    int denom = nc::sqrt(nc::power(Gxy, 2) + nc::power((Gxx - Gyy), 2)) + nc::finfo(float).eps;

    auto sin2theta = Gxy / denom; //Sine and cosine of doubled angles
    auto cos2theta = (Gxx - Gyy) / denom;

    if (orientsmoothsigma)
    {
        sze = nc::fix(6 * orientsmoothsigma);
        if (nc::remainder(sze, 2) == 0)
        {
            sze = nc::fix(6 * orientsmoothsigma);
            if (nc::remainder(sze, 2) == 0)
            {
                sze = sze + 1;
            }
            gauss = cv2.getGaussianKernel(nc::int(sze), orientsmoothsigma);
            f = gauss * gauss.T;
            cos2theta = ndimage.convolve(cos2theta, f); //Smoothed sine and cosine of
            sin2theta = ndimage.convolve(sin2theta, f); // doubled angles
        }
    }

    auto orientim = nc::pi / 2 + nc::arctan2(sin2theta, cos2theta) / 2;
    return (orientim);
}

auto frequest(auto im, auto orientim, auto windsze, auto minWaveLength, auto maxWaveLength)
{
    auto cols = im.shape;
    auto rows = im.shape;

    // # Find mean orientation within the block. This is done by averaging the
    // # sines and cosines of the doubled angles before reconstructing the
    // # angle again.  This avoids wraparound problems at the origin.

    auto cosorient = nc::mean(nc::cos(2 * orientim));
    auto sinorient = nc::mean(nc::sin(2 * orientim));
    auto orient = math.atan2(sinorient, cosorient) / 2;

    // # Rotate the image block so that the ridges are vertical

    // # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/nc::pi*180 + 90,1)
    // # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))

    auto rotim = scipy.ndimage.rotate(im, orient / nc::pi * 180 + 90, axes = (1, 0), reshape = False, order = 3, mode = 'nearest');

    // # Now crop the image so that the rotated image does not contain any
    // # invalid regions.  This prevents the projection down the columns
    // # from being mucked up.

    auto cropsze = int(nc::fix(rows / nc::sqrt(2)));
    auto offset = int(nc::fix((rows - cropsze) / 2));
    auto rotim = rotim [offset:offset + cropsze][:, offset:offset + cropsze];

    // # Sum down the columns to get a projection of the grey values down
    // # the ridges.

    auto proj = nc::sum(rotim, nc::axis::ROW);  //nc::sum(rotim, axis = 0);
    auto dilation = scipy.ndimage.grey_dilation(proj, windsze, structure = nc::ones(windsze));

    auto temp = nc::abs(dilation - proj);

    auto peak_thresh = 2;

    auto maxpts = (temp < peak_thresh) & (proj > nc::mean(proj));
    auto maxind = nc::where(maxpts);

    auto rows_maxind, cols_maxind = nc::shape(maxind);

    // # Determine the spatial frequency of the ridges by divinding the
    // # distance between the 1st and last peaks by the (No of peaks-1). If no
    // # peaks are detected, or the wavelength is outside the allowed bounds,
    // # the frequency image is set to 0

    if (cols_maxind < 2)
    {

        auto freqim = nc::zeros(im.shape);
    }
    else
    {

        auto NoOfPeaks = cols_maxind;
        auto waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1);
        if (waveLength >= minWaveLength && waveLength <= maxWaveLength)
        {
            freqim = 1 / nc::double(waveLength) * nc::ones(im.shape);
        }

        else
        {
            freqim = nc::zeros(im.shape);
        }
    }
    return (freqim);
}

auto ridge_freq(auto im, auto mask, auto orient, auto blksze, auto windsze, auto minWaveLength, auto maxWaveLength)
{
    auto cols = im.shape;
    auto rows = im.shape;

    for (int r = 0, rows - blksze, blksze)
    {
        for (int c = 0, cols - blksze, blksze)
        {
            blkim = im [r:r + blksze][:, c:c + blksze];
            blkor = orient [r:r + blksze][:, c:c + blksze];

            freq [r:r + blksze][:, c:c + blksze] = frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength);
        }
    }
    auto freq = freq * mask;
    auto freq_1d = nc::reshape(freq, (1, rows * cols));
    auto ind = nc::where(freq_1d > 0);

    nc::array ind = (ind);  //ind = nc::array (ind)
    ind = ind [1, :];

    non_zero_elems_in_freq = freq_1d[0][ind];

    auto meanfreq = nc::mean(non_zero_elems_in_freq);
    auto medianfreq = nc::median(non_zero_elems_in_freq); //# does not work properly
    return freq, meanfreq;
}

auto ridge_filter(auto im, auto orient, auto freq, auto kx, auto ky)
{

    auto angleInc = 3;
    auto im = nc::double(im);
    auto rows, cols = im.shape;
    auto newim = nc::zeros((rows, cols));

    auto req_1d = nc::reshape(freq, (1, rows * cols));
    auto ind = nc::where(freq_1d > 0);

    ind = nc::array(ind);
    ind = ind [1, :];

    // # Round the array of frequencies to the nearest 0.01 to reduce the
    // # number of distinct frequencies we have to deal with.

    auto non_zero_elems_in_freq = freq_1d[0][ind];
    non_zero_elems_in_freq = nc::double(nc::round((non_zero_elems_in_freq * 100))) / 100;

    auto unfreq = nc::unique(non_zero_elems_in_freq);

    // # Generate filters corresponding to these distinct frequencies and
    // # orientations in 'angleInc' increments.

    auto sigmax = 1 / unfreq[0] * kx;
    auto sigmay = 1 / unfreq[0] * ky;

    auto sze = nc::round(3 * nc::max{([ sigmax, sigmay ]}));///look into this

    auto x, y = nc::meshgrid(nc::linspace(-sze, sze, (2 * sze + 1)), nc::linspace(-sze, sze, (2 * sze + 1)));

    auto reffilter = nc::exp(-((nc::power(x, 2)) / (sigmax * sigmax) + (nc::power(y, 2)) / (sigmay * sigmay))) * nc::cos(2 * nc::pi * unfreq[0] * x);
    //#this is the original gabor filter

    auto filt_rows, filt_cols = reffilter.shape;
    auto filt_cols = reffilter.shape;
    auto gabor_filter = nc::array(nc::zeros((int(180 / angleInc), int(filt_rows), int(filt_cols))));

    //#Generate rotated versions of the filter.Note orientation
    // #image provides orientation *along *the ridges, hence + 90
    // #degrees, and imrotate requires angles + ve anticlockwise, hence
    // #the minus sign.

    for( int o = 0 ,  (0, int(180 / angleInc){
        auto rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape = False);
        gabor_filter[o] = rot_filt;

    }
                                             
// #Find indices of matrix points greater than maxsze from the image
// #boundary

    auto maxsze = int(sze);

    auto temp = freq > 0;
    auto validr, validc = nc::where(temp)
    auto temp1 = validr > maxsze;
    auto temp2 = validr < rows - maxsze;
    auto temp3 = validc > maxsze;
    auto temp4 = validc < cols - maxsze;

    auto final_temp = temp1 & temp2 & temp3 & temp4;

    auto finalind = nc::where(final_temp);

// #Convert orientation matrix values from radians to an index value
// #that corresponds to round(degrees / angleInc)

    auto maxorientindex = nc::round(180 / angleInc);
    auto orientindex = nc::round(orient / nc::pi * 180 / angleInc);

// #do the filtering


 for (int i = 0, (0, rows))
    {
        for (int j = 0, (0, cols))
        {
            if (orientindex[i][j] < 1)
            {
                orientindex[i][j] = orientindex[i][j] + maxorientindex;
            }
        }
    }
if (orientindex[i][j] > maxorientindex){
        orientindex[i][j] = orientindex[i][j] - maxorientindex;
}
       auto finalind_rows, finalind_cols = nc::shape(finalind);
    auto sze = int(sze);          
    
    for (int k = 0 , (0, finalind_cols)){
        auto r = validr[finalind[0][k]];
        auto c = validc[finalind[0][k]];
    }
       
    auto img_block = im [r - sze:r + sze + 1][:, c - sze:c + sze + 1];

    auto newim[r][c] = nc::sum(img_block * gabor_filter[int(orientindex[r][c]) - 1]);

    return (newim);
}

auto image_enhance(auto img)
{
    blksze = 16;
    thresh = 0.2;
    normim, mask = ridge_segment(img, blksze, thresh); // # normalise the image and find a ROI

    orientsmoothsigma = 7;
    gradientsigma = 1;
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma); //# find orientation of every pixel
    blocksigma = 7;
    blksze = 38;

    windsze = 5;
    auto minWaveLength = 5;
    auto maxWaveLength = 15;
    auto freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength, maxWaveLength); //# find the overall frequency of ridges

    freq = medfreq * mask;
    auto kx = 0.65;
    auto ky = 0.65;
    auto newim = ridge_filter(normim, orientim, freq, kx, ky); // # create gabor filter and do the actual filtering

    //# th, bin_im = cv2.threshold(np.uint8(newim),0,255,cv2.THRESH_BINARY);
    return (newim < -3);
}

auto removedot(invertThin)
{
    temp0 = np.array(invertThin[:])
                temp0 = np.array(temp0)
                            temp1 = temp0 / 255 temp2 = np.array(temp1)
                                                            temp3 = np.array(temp2)

                                                                        enhanced_img = np.array(temp0)
                                                                                           filter0 = np.zeros((10, 10))
                                                                                                         W,
    H = temp0.shape[:2] filtersize = 6 for (int i = 0, (W - filtersize))
    {
        for (int j = 0, (H - filtersize))
        {
            filter0 = temp1 [i:i + filtersize, j:j + filtersize]

                flag = 0;
            if
                sum(filter0[:, 0]) == 0 : flag += 1;
            if
                sum(filter0[:, filtersize - 1]) == 0 : flag += 1;
            if
                sum(filter0 [0, :]) == 0 : flag += 1;
            if
                sum(filter0 [filtersize - 1, :]) == 0 : flag += 1;
            if
                flag > 3 :
                    temp2 [i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize));
        }
    }

    return temp2;
}

auto get_descriptors(img)
{
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8));
    img = clahe.apply(img);
    img = image_enhance(img);
    img = np.array(img, dtype = np.uint8);
    //#Threshold
    img = cv2.equalizeHist(img);
    auto ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU);
    //#cv2.imshow("Hello", img)
    img = img [246:1570, 0:1680];
    img = img [0:1570, 150:1680];
    cv2.imshow("New", img);
    cv2.imwrite("/Users/simonakpoveso/Downloads/new12.bmp", img);

    auto img1 = img [230:810, 340:790];
    cv2.imshow("Cropped", img1);
    cv2.imwrite("/Users/simonakpoveso/Downloads/cropnew12.bmp", img1);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
    //Normalize to 0 and 1 range
    img[img == 255] = 1;

    //#Thinning
    auto skeleton = skeletonize(img);
    skeleton = np.array(skeleton, dtype = np.uint8);
    skeleton = removedot(skeleton);
    // #plt.imshow(skeleton)
    // #plt.show()

    // #plt.imshow(img)
    // #plt.show()
    // #Harris corners
    auto harris_corners = cv2.cornerHarris(img, 5, 5, 0.04);
    auto harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32FC1);
    auto threshold_harris = 150;
    //#Extract keypoints
    auto keypoints = [];
    for (int x = 0, (0, harris_normalized.shape[0]))
    {
        for (int j = 0, (0, harris_normalized.shape[1]))
        {
            if (harris_normalized[x][y] > threshold_harris)
            {
                keypoints.append(cv2.KeyPoint(y, x, 1));
            }
        }
    }
    //#Define descriptor
    auto orb = cv2.ORB_create()
               //Compute descriptors
               _,
         auto des = orb.compute(img, keypoints) ///////////review
                    return keypoints,
         des;
}

void main(void)
{
    //#Take the picture
    // #camera = picamera.PiCamera()  # initialize camera
    // #camera.resolution = (3280, 2464)
    // #camera.color_effects = (128, 128)
    // #camera.capture('image' + count + '.bmp')

    // # Read Image
    char image_name = "enirladitest.bmp";
    img1 = cv2.imread("Database/" + image_name, cv2.IMREAD_GRAYSCALE);
    //# Rotate the Image
    auto angle = -90;
    auto h, w = img1.shape[:2];
    //# calculate center of Image
    auto center = (w / 2, h / 2);
    //# Rotate
    auto M = cv2.getRotationMatrix2D(center, angle, 1.0);
    img1 = cv2.warpAffine(img1, M, (w, h));

    //#cv2.imwrite("/Users/simonakpoveso/Downloads/bin1.bmp", img1)

    point1 = np.float32([ [ 0, 0 ], [ 2050, 0 ], [ 2050, 2050 ], [ 0, 2050 ] ]);
    point2 = np.float32([ [ 212, -312 ], [ 2076, 420 ], [ 1828, 1688 ], [ -36, 1596 ] ]);

    // #cv2.circle(img1, (0, 246), 10, (255, 0, 255), -1)  # Top Left
    // #cv2.circle(img1, (1680, 246), 10, (255, 0, 255), -1)  # Top Right
    // #cv2.circle(img1, (1680, 1570), 10, (255, 0, 255), -1)  # Bottom Right
    // #cv2.circle(img1, (0, 1570), 10, (255, 0, 255), -1)  # Bottom Left

    auto matrix = cv2.getPerspectiveTransform(point1, point2);
    img1 = cv2.warpPerspective(img1, matrix, (1700, 1680));

    //#Crop Image again to remove unnecessary binarization

    // #cv2.imshow("Them", img1)
    // #cv2.waitKey(0)
    auto kp1, des1 = get_descriptors(img1);
    //#img1 = img1[246:1570, 0:1680]

    image_name = "enileft.bmp";
    img2 = cv2.imread("Database/" + image_name, cv2.IMREAD_GRAYSCALE);
    kp2, des2 = get_descriptors(img2);

    //# Matching between descriptors
    auto bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True);
    auto matches = sorted(bf.match(des1, des2), key = lambda match
                          : match.distance);
    //# Plot keypoints
    img4 = cv2.drawKeypoints(img1, kp1, outImage = None);
    img5 = cv2.drawKeypoints(img2, kp2, outImage = None);
    auto f, axarr = plt.subplots(1, 2);
    auto axarr[0].imshow(img4);
    auto axarr[1].imshow(img5);
    plt.show()
        //# Plot matches
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags = 2, outImg = None)
                   plt.imshow(img3)
                       plt.show()

    // # Calculate score
    //  #score = 0
    //  #for match in matches:
    //  #   score += match.distance
    //  #   score_threshold = 20
    // #if score / len(matches) < score_threshold:
    // #    print("Fingerprint matches.")
    // #    print((score / len(matches)))
    // #else:
    // #    print("Fingerprint does not match.")
    // #    print((score / len(matches)))
}
