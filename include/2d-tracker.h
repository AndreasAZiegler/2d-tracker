#ifndef MARKERTRACKER_H
#define MARKERTRACKER_H

// Includes
#include <Eigen/Dense>
#include <aruco/markerdetector.h>
#include <chrono>

/**
 * @brief Kalman filter which tracks fiducial markers in 2D. Can be used to
 * 				restrict the area(s) in which a marker detecter has to search for
 * 				markers in an image.
 */
class MarkerTracker
{
  public:
    /**
     * @brief Constructor (empty)
     */
    MarkerTracker();

    /**
     * @brief Destructor (empty)
     */
    ~MarkerTracker();

    /**
     * @brief Initializes Kalman Filter with the given markers.
     * @param markers Markers with which the Kalman Filter will be initialized.
     */
    void initializeMarkers(std::vector<aruco::Marker> markers);

    /**
     * @brief Adds markers to the Kalman Filter.
     * @param markers Markers to add.
     */
    void addMarkers(const std::vector<aruco::Marker> markers);

    /**
     * @brief Removes a marker from the Kalman Filter.
     * @param markerID Id of the marker which should be removed.
     */
    void removeMarker(const int markerID);

    /**
     * @brief Predicts the center of the marker in the next frame.
     */
    void predict();

    /**
     * @brief Get the predicted location of the markers.
     * @return Vector with the id and the location of the predicted markers.
     */
    std::vector<std::tuple<int, cv::Point2d> > getPredictedMarkers();

    /**
     * @brief Updates the Kalman Filter (2nd step)
     * @param markers Observed markers (used as measurement)
     * @return Current state (markers).
     */
    Eigen::Matrix<double, Eigen::Dynamic, 1> update(std::vector<aruco::Marker> markers);

  private:
    Eigen::Matrix<double, Eigen::Dynamic, 1> state; /**< The state (marker id, y- and x-location and velocities) */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Pp; /**< Predicted covariances of the states */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Pm; /**< Updated covariances of the states */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q; /**< Covariance of the process noise */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R; /**< Covariance of the sensor noise */
    std::chrono::system_clock::time_point last_update; /**< Time at the last prediction */
};

#endif // MARKERTRACKER_H
