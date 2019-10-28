
auto stddevim;
auto im, blksze , thresh, std;
auto img;
#ifndef _MAIN_H_
#define _MAIN_H_



auto mean;
auto minWaveLength;
auto maxWaveLength;
auto peak_thresh;
auto normim, mask;
auto gradientsigma;
auto blocksigma;
auto orientsmoothsigma;
auto stddevim;
auto minWaveLength;
auto maxWaveLength;
auto invertThin;
auto temp0, temp1, temp2, temp3;
auto enhanced_img;
auto filter0;
auto W, H;
auto filtersize;
auto clahe;
auto normalise(auto img, auto mean, auto std);
auto point1, point2;
auto kp1, des1, img1;
auto kp2, des2, img2, img3, img4, img5;
auto ridge_segment(auto im, auto blksze, auto thresh);

auto ridge_filter(auto im, auto orient, auto freq, auto kx, auto ky);
auto image_enhance(auto img);

auto ridge_freq(auto im, auto mask, auto orient, auto blksze, auto windsze, auto minWaveLength, auto maxWaveLength);

auto frequest(auto im, auto orientim, auto windsze, auto minWaveLength, auto maxWaveLength);

auto ridge_orient(auto im, auto gradientsigma, auto blocksigma, auto orientsmoothsigma);
auto get_descriptors(auto img);


#endif