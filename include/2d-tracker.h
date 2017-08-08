#pragma once

// Includes
#include <Eigen/Dense>
#include <aruco/markerdetector.h>
#include <chrono>

/**
 * @brief Kalman filter which tracks fiducial markers in 2D. Can be used to
 * 				restrict the area(s) in which a marker detecter has to search
 * for
 * 				markers in an image.
 */
class MarkerTracker
{
public:
    MarkerTracker();

    ~MarkerTracker();

    /**
     * @brief Initializes Kalman Filter with the given markers.
     * @param markers Markers with which the Kalman Filter will be initialized.
     */
    void initializeMarkers(std::vector<aruco::Marker> markers);

    /**
     * @brief Returns if the Kalman filter is initialized or not.
     * @return true if Kalman filter is initialized, false otherwise.
     */
    bool isInitialized();

    /**
     * @brief Sets the initialized flag according to passed variable.
     * @param initialized true if Kalman filter should be set initialized, false
     * otherwise.
     */
    void setInitializedFlag(bool initialized);

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
    std::vector<std::tuple<int, cv::Point2d>> getPredictedMarkers();

    /**
     * @brief Updates the Kalman Filter (2nd step)
     * @param markers Observed markers (used as measurement)
     * @return Current state (markers).
     */
    Eigen::Matrix<double, Eigen::Dynamic, 1> update(
        std::vector<aruco::Marker> markers);

    /**
     * @brief Returns the current states and covariances
     * @return The current states and covariances
     */
    std::string prettyPrint();

private:
    /**
     * @brief Checks if the id belongs to a tracked marker.
     * @param id Id of the marker which should be checked if it is tracked.
     * @return True if the marker is tracked, False if the marker is not tracked
     */
    bool isMarkerTracked(int id);

private:
    Eigen::VectorXd
        m_state; /**< The state (marker id, y- and x-location and velocities) */
    Eigen::MatrixXd m_Pp; /**< Predicted covariances of the states */
    Eigen::MatrixXd m_Pm; /**< Updated covariances of the states */
    Eigen::MatrixXd m_Q;  /**< Covariance of the process noise */
    Eigen::MatrixXd m_R;  /**< Covariance of the sensor noise */
    Eigen::MatrixXd m_K;  /**< Kalman gain matrix */
    std::chrono::system_clock::time_point
        m_last_update;      /**< Time at the last prediction */
    bool m_InitializedFlag; /**< Indicates if the Kalman filter is initialized
                               or not */
    const int NUM_STATE_VARS_PER_MARKER =
        5; /**< Number of state variables a marker represents */
};
