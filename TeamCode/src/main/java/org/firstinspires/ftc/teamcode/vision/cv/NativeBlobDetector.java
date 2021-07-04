package org.firstinspires.ftc.teamcode.vision.cv;

import org.firstinspires.ftc.teamcode.vision.cv.CvUtils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.SimpleBlobDetector;

public class NativeBlobDetector {

    SimpleBlobDetector blobDet;

    public NativeBlobDetector(double minArea, double minCircularity, double maxCircularity) {
        blobDet = CvUtils.cvCreateBlobDetector(minArea, new double[] { minCircularity, maxCircularity}, false);
    }

    public MatOfKeyPoint detect(Mat input) {
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        blobDet.detect(input, keypoints);
        return keypoints;
    }

    public void release() {
        blobDet.clear();
    }

}
