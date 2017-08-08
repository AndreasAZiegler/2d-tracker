// Includes
#include "2d-tracker.h"
#include <Eigen/Core>

// Output format definition for Eigen to print in Octave/Matlab style.
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[",
                          "]");

MarkerTracker::MarkerTracker() {}

MarkerTracker::~MarkerTracker() {}

/**
 * @brief Initializes Kalman Filter with the given markers.
 * @param markers Markers with which the Kalman Filter will be initialized.
 */
void MarkerTracker::initializeMarkers(std::vector<aruco::Marker> markers) {
  int numStates = NUM_STATE_VARS_PER_MARKER * markers.size();

  // Initializes states with the marker id, the location and velocity (= 0)
  m_state.resize(numStates, 1);
  for (int i = 0; i < markers.size(); i++) {
      m_state.block<5, 1>(i * NUM_STATE_VARS_PER_MARKER, 0) << markers[i].id,
          markers[i].getCenter().y, markers[i].getCenter().x, 0, 0;
  }

  // Initialize covariances of the predicted- and updated states with zero
  m_Pp.resize(numStates, numStates);
  m_Pm.setZero();

  m_Pm.resize(numStates, numStates);
  m_Pm.setZero();

  // Initialize covariances of the process model
  m_Q.resize(numStates, numStates);
  m_Q.setIdentity();
  m_Q *= 50;

  // Initialize covariances of the sensor
  m_R.resize(numStates, numStates);
  m_R.setIdentity();
  m_R *= 5;

  // Initialize Kalman gain matrix
  m_K.resize(numStates, numStates);
  m_K.setZero();

  // Initialize time point
  m_last_update = std::chrono::system_clock::now();

  m_InitializedFlag = true;
}

/**
 * @brief Returns if the Kalman filter is initialized or not.
 * @return true if Kalman filter is initialized, false otherwise.
 */
bool MarkerTracker::isInitialized() {
  return m_InitializedFlag;
}

/**
 * @brief Sets the initialized flag according to passed variable.
 * @param initialized true if Kalman filter should be set initialized, false
 * otherwise.
 */
void MarkerTracker::setInitializedFlag(bool initialized) {
  m_InitializedFlag = initialized;
}

/**
 * @brief Adds a marker to the Kalman Filter.
 * @param markers Marker to add.
 * @todo Proper implementation
 */
void MarkerTracker::addMarkers(const std::vector<aruco::Marker> markers) {
  std::vector<aruco::Marker> markers_filtered;

  // Return if the markers vector is empty
  if(markers.empty()) {
    return;
  }

  int numTrackedStates = m_state.rows();

  // Only add markers which are not yet tracked
  // bool add;
  for(const aruco::Marker& marker : markers) {
    if(true == isMarkerTracked(marker.id)) {
      markers_filtered.push_back(marker);
    }
  }

  int nrOfMarkers = markers_filtered.size();

  // Resize/extend states
  m_state.conservativeResize(numTrackedStates +
                             nrOfMarkers * NUM_STATE_VARS_PER_MARKER);

  int numTrackedStatesNew = m_state.rows();
  int numNewTrackedStates = numTrackedStatesNew - numTrackedStates;
  int numNewTrackedMarkers = numNewTrackedStates / NUM_STATE_VARS_PER_MARKER;
  // Fill in the new states
  for (int i = 0; i < numNewTrackedMarkers; i++) {
    m_state.block<5, 1>(numTrackedStates + i * NUM_STATE_VARS_PER_MARKER, 0)
        << markers_filtered[i].id,
        markers_filtered[i].getCenter().y,
        markers_filtered[i].getCenter().x, 0, 0;
  }

  // Fill in 0 in the new covariance entries
  m_Pp.conservativeResize(
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER,
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER);
  m_Pm.conservativeResize(
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER,
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER);
  m_Q.conservativeResize(
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER,
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER);
  m_R.conservativeResize(
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER,
      numTrackedStates + nrOfMarkers * NUM_STATE_VARS_PER_MARKER);

  m_Pp.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_Pp.block(numTrackedStates, 0, numNewTrackedStates, numTrackedStatesNew)
      .setZero();

  m_Pm.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_Pm.block(numTrackedStates, 0, numNewTrackedStates, numTrackedStatesNew)
      .setZero();

  m_Q.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_Q.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_Q.block(numTrackedStates, 0, numNewTrackedStates, numTrackedStates)
      .setZero();
  m_Q.block(numTrackedStates, numTrackedStates, numNewTrackedStates,
          numNewTrackedStates) =
      50 *
      Eigen::MatrixXd::Identity(numNewTrackedStates, numNewTrackedStates);

  m_R.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_R.block(0, numTrackedStates, numTrackedStates, numNewTrackedStates)
      .setZero();
  m_R.block(numTrackedStates, 0, numNewTrackedStates, numTrackedStates)
      .setZero();
  m_R.block(numTrackedStates, numTrackedStates, numNewTrackedStates,
            numNewTrackedStates) =
      5 * Eigen::MatrixXd::Identity(numNewTrackedStates, numNewTrackedStates);
}

/**
 * @brief Removes a marker from the Kalman Filter.
 * @param markerID Id of the marker which should be removed.
 */
void MarkerTracker::removeMarker(const int markerID) {
  // Return if marker is invalid
  if(0 > markerID) {
    return;
  }

  // Return if marker id does not belong to a tracked marker.
  if(false == isMarkerTracked(markerID)) {
    return;
  }

  int numTrackedStatesOld = m_state.rows();

  // Save current state and covariances
  Eigen::VectorXd state_tmp = m_state;
  Eigen::MatrixXd Pp_tmp = m_Pp;
  Eigen::MatrixXd Pm_tmp = m_Pm;
  Eigen::MatrixXd Q_tmp = m_Q;
  Eigen::MatrixXd R_tmp = m_R;
  int numTrackedStatesNew = numTrackedStatesOld - NUM_STATE_VARS_PER_MARKER;

  // Set states and covariances to 0
  m_state.setZero(numTrackedStatesNew, 1);
  m_Pp.setZero(numTrackedStatesNew, numTrackedStatesNew);
  m_Pm.setZero(numTrackedStatesNew, numTrackedStatesNew);
  m_Q.setZero(numTrackedStatesNew, numTrackedStatesNew);
  m_R.setZero(numTrackedStatesNew, numTrackedStatesNew);

  // Copy values from the saved states/covariances to the new ones and skip
  // the ones
  // which should be deleted
  for(int i = 0, ii = 0; i < numTrackedStatesNew;
      i += NUM_STATE_VARS_PER_MARKER, ii += NUM_STATE_VARS_PER_MARKER) {
    if(markerID == state_tmp[i]) {
      ii += NUM_STATE_VARS_PER_MARKER;
    }

    m_state.block<5, 1>(i, 0) = state_tmp.block<5, 1>(ii, 0);
  }

  for(int i = 0, ii = 0; i < numTrackedStatesNew; i++, ii++) {
    if(0 == (i % NUM_STATE_VARS_PER_MARKER)) {
      if(markerID == state_tmp[i]) {
        ii += NUM_STATE_VARS_PER_MARKER;
      }
    }
    for(int j = 0, jj = 0; j < numTrackedStatesNew; j++, jj++) {
      if(0 == (j % NUM_STATE_VARS_PER_MARKER)) {
        if(markerID == state_tmp[j]) {
          jj += NUM_STATE_VARS_PER_MARKER;
        }
      }
      m_Pp(i, j) = Pp_tmp(ii, jj);
      m_Pm(i, j) = Pm_tmp(ii, jj);
      m_Q(i, j) = Q_tmp(ii, jj);
      m_R(i, j) = R_tmp(ii, jj);
    }
  }
}

/**
 * @brief Predicts the center of the marker in the next frame.
 */
void MarkerTracker::predict() {
  // Get current time and calculate time difference since last prediction
  std::chrono::system_clock::time_point time_now =
      std::chrono::system_clock::now();
  std::chrono::duration<double> time_diff = time_now - m_last_update;

  const int numTrackedStates = m_state.rows();

  // Create system matrix A
  // (Identity matrix plus \delta_t for position ({x,y}_new = {x,y}_old +
  // {v_x,v_y}_old * \delta_t)
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
  A = Eigen::MatrixXd::Identity(numTrackedStates, numTrackedStates);
  for(int i = 0; i < numTrackedStates; i += NUM_STATE_VARS_PER_MARKER) {
    A(i + 1, i + 3) = time_diff.count();
  }

  // Predict new states
  m_state = A * m_state;

  // Predict new covarinces of the states
  m_Pp = A * m_Pm * A.transpose() + m_Q;

  // Update time point
  m_last_update = time_now;
}

/**
 * @brief Get the predicted location of the markers.
 * @return Vector with the id and the location of the predicted markers.
 */
std::vector<std::tuple<int, cv::Point2d>> MarkerTracker::getPredictedMarkers() {
  int numTrackedMarkers = m_state.rows() / NUM_STATE_VARS_PER_MARKER;
  std::vector<std::tuple<int, cv::Point2d>> predictedPoints(
      numTrackedMarkers);

  // Create vector of tuples containing marker id and position
  for(int i = 0; i < numTrackedMarkers; i++) {
    std::get<0>(predictedPoints[i]) =
        m_state(i * NUM_STATE_VARS_PER_MARKER + 0, 0);
    std::get<1>(predictedPoints[i]).y =
        m_state(i * NUM_STATE_VARS_PER_MARKER + 1, 0);
    std::get<1>(predictedPoints[i]).x =
        m_state(i * NUM_STATE_VARS_PER_MARKER + 2, 0);
  }

  return predictedPoints;
}

/**
 * @brief Updates the Kalman Filter (2nd step)
 * @param markers Observed markers (used as measurement)
 * @return Current state (markers).
 */
Eigen::Matrix<double, Eigen::Dynamic, 1> MarkerTracker::update(
    std::vector<aruco::Marker> markers) {
  // Check if detected markers are within the tracked markers.
  // Delete tracked markers which were not detected.
  bool found;
  for(int i = 0; i < m_state.rows(); i += NUM_STATE_VARS_PER_MARKER) {
    found = false;
    for(const aruco::Marker& marker : markers) {
      if(0 > marker.id) {
        continue;
      }
      if(marker.id == static_cast<int>(m_state(i))) {
        found = true;
        continue;
      }
    }

    if(false == found) {
      removeMarker(m_state(i));
      i -= NUM_STATE_VARS_PER_MARKER;
    }
  }

  // Return if state is empty
  if(0 == m_state.rows()) {
    return (m_state);
  }

  // Create measurement vector
  int numTrackedStates = m_Pp.rows();
  Eigen::Matrix<double, Eigen::Dynamic, 1> measurements;
  measurements.setZero(numTrackedStates, 1);

  // Set measurement values from the given markers and set velocity to 0
  // Temporary save markers which were detected but not yet tracked
  // to add them after the update step
  std::vector<aruco::Marker> markersToAdd;
  for(const aruco::Marker& marker : markers) {
    found = false;
    for(int i = 0; i < numTrackedStates; i += NUM_STATE_VARS_PER_MARKER) {
      if(marker.id == static_cast<int>(m_state(i))) {
        measurements.block<5, 1>(i, 0) << marker.id,
            marker.getCenter().y, marker.getCenter().x, 0, 0;
        found = true;
        continue;
      }
    }
    if(false == found) {
      markersToAdd.push_back(marker);
    }
  }
  numTrackedStates = m_Pp.rows();

  // measurements.format(OctaveFmt));
  // Create Kalman gain matrix (K)
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
  m_K.resize(numTrackedStates, numTrackedStates);

  // Calculate PR (The inverse is later used)
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PR_inv = m_Pp + m_R;

  // Calculate the inverse
  PR_inv = PR_inv.inverse();

  m_K = m_Pp * PR_inv;

  // Create error vector
  Eigen::Matrix<double, Eigen::Dynamic, 1> error = measurements - m_state;
  // Update states with the Kalman gain matrix and the error
  m_state = m_state + m_K * error;

  // Update the covariances of the states with the Kalman gain matrix and the
  // error
  m_Pm =
      (Eigen::MatrixXd::Identity(numTrackedStates, numTrackedStates) - m_K) *
      m_Pp;

  // Add detected markers which are not yet tracked
  if(false == markersToAdd.empty()) {
    addMarkers(markersToAdd);
  }

  // Update time point
  m_last_update = std::chrono::system_clock::now();

  return m_state;
}

/**
 * @brief Returns the current states and covariances
 * @return The current states and covariances
 */
std::string MarkerTracker::prettyPrint() {
  std::stringstream output;
  output << "State: \n"
         << m_state << "\n\nPp: \n"
         << m_Pp << "\n\nPm: \n"
         << m_Pm << "\n\nQ: \n"
         << m_Q << "\n\nR: \n"
         << m_R << "\n\nK: \n"
         << m_K << std::endl;
  return output.str();
}

/**
 * @brief Checks if the id belongs to a tracked marker.
 * @param id Id of the marker which should be checked if it is tracked.
 * @return True if the marker is tracked, False if the marker is not tracked
 */
bool MarkerTracker::isMarkerTracked(int id) {
  // Return if id is invalid
  if(0 > id) {
    return (false);
  }

  int numTrackedStates = m_state.size();
  for(int i = 0; i < numTrackedStates; i += NUM_STATE_VARS_PER_MARKER) {
    if(id == m_state(i, 0)) {
      return (true);
    }
  }
  return false;
}
