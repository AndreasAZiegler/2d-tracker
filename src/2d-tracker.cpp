// Includes
#include "../include/2d-tracker.h"
#include <Eigen/Core>

// Output format definition for Eigen to print in Octave/Matlab style.
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

/**
 * @brief Constructor (empty)
 */
MarkerTracker::MarkerTracker() {}

/**
 * @brief Destructor (empty)
 */
MarkerTracker::~MarkerTracker() {}

/**
 * @brief Initializes Kalman Filter with the given markers.
 * @param markers Markers with which the Kalman Filter will be initialized.
 */
void MarkerTracker::initializeMarkers(std::vector<aruco::Marker> markers) {
  int nrOfMarkers = markers.size();

  // Initializes states with the marker id, the location and velocity (= 0)
  state.resize(5 * nrOfMarkers, 1);
  for(int i = 0; i < markers.size(); i++) {
    state(i * 5 + 0) = markers[i].id;
    state(i * 5 + 1) = markers[i].getCenter().y;
    state(i * 5 + 2) = markers[i].getCenter().x;
    state(i * 5 + 3) = 0;
    state(i * 5 + 4) = 0;
  }

  // Initialize covariances of the predicted- and updated states with zero
  Pp.resize(5 * nrOfMarkers, 5 * nrOfMarkers);
  Pm.setZero();

  Pm.resize(5 * nrOfMarkers, 5 * nrOfMarkers);
  Pm.setZero();

  // Initialize covariances of the process model
  Q.resize(5 * nrOfMarkers, 5 * nrOfMarkers);
  Q = 50 * Eigen::MatrixXd::Identity(5 * nrOfMarkers, 5 * nrOfMarkers);

  // Initialize covariances of the sensor
  R.resize(5 * nrOfMarkers, 5 * nrOfMarkers);
  R = 5 * Eigen::MatrixXd::Identity(5 * nrOfMarkers, 5 * nrOfMarkers);

  // Initialize time point
  last_update = std::chrono::system_clock::now();

  /*
  ROS_INFO_STREAM("Initialization: state: " << state.format(OctaveFmt));
  ROS_INFO_STREAM("Pm: " << Pm.format(OctaveFmt));
  ROS_INFO_STREAM("Q: " << Q.format(OctaveFmt));
  ROS_INFO_STREAM("R: " << R.format(OctaveFmt));
  */
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

	int rows = state.rows();
	int rows_5 = rows / 5;

	// Only add markers which are not yet tracked
	bool add;
	for(const aruco::Marker& marker : markers) {
		add = true;
		for(int i = 0; i < rows_5; i += 5) {
			if(marker.id == state(i))	{
				add = false;
			}
		}
		if(true == add) {
			markers_filtered.push_back(marker);
		}
	}

	int nrOfMarkers = markers_filtered.size();

	//ROS_INFO_STREAM("addMarkers: state_old:\n" << state);

	// Resize/extend states
	state.conservativeResize(rows + nrOfMarkers * 5);

	int new_rows = state.rows();
	int row_diff = new_rows - rows;
	int row_diff_5 = row_diff / 5;
	// Fill in the new states
	for(int i = 0; i < row_diff_5; i++) {
		state(rows + i*5 + 0) = markers_filtered[i].id;
		state(rows + i*5 + 1) = markers_filtered[i].getCenter().y;
		state(rows + i*5 + 2) = markers_filtered[i].getCenter().x;
		state(rows + i*5 + 3) = 0;
		state(rows + i*5 + 4) = 0;
	}
	/*
	ROS_INFO_STREAM("addMarkers: state_new:\n" << state);

	ROS_INFO_STREAM("addMarkers: Pp_old:\n" << Pp);
	ROS_INFO_STREAM("addMarkers: Pm_old:\n" << Pm);
	ROS_INFO_STREAM("addMarkers: Q_old:\n" << Q);
	ROS_INFO_STREAM("addMarkers: R_old:\n" << R);
	*/

	// Fill in 0 in the new covariance entries
	Pp.conservativeResize(rows + nrOfMarkers * 5, rows + nrOfMarkers * 5);
	Pm.conservativeResize(rows + nrOfMarkers * 5, rows + nrOfMarkers * 5);
	Q.conservativeResize(rows + nrOfMarkers * 5, rows + nrOfMarkers * 5);
	R.conservativeResize(rows + nrOfMarkers * 5, rows + nrOfMarkers * 5);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < row_diff; j++) {
			Pp(i, rows + j) = 0;
			Pm(i, rows + j) = 0;

			Q(i, rows + j) = 0;
			R(i, rows + j) = 0;
		}
	}
	for(int i = rows; i < new_rows; i++) {
		for(int j = 0; j < new_rows; j++) {
			Pp(i, j) = 0;
			Pm(i, j) = 0;

			if(i == j) {
				Q(i, j) = 50;
				R(i, j) = 5;
			} else {
				Q(i, j) = 0;
				R(i, j) = 0;
			}
		}
	}

	/*
	ROS_INFO_STREAM("addMarkers: Pp_new:\n" << Pp);
	ROS_INFO_STREAM("addMarkers: Pm_new:\n" << Pm);
	ROS_INFO_STREAM("addMarkers: Q_new:\n" << Q);
	ROS_INFO_STREAM("addMarkers: R_new:\n" << R);
	*/

	/*
	ROS_INFO_STREAM("addMarker: state: " << state.format(OctaveFmt));
	*/
	std::cout << "addMarker: added marker(s)" << std::endl;
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

  int old_rows = state.rows();

	// Return if marker id does not belong to a tracked marker.
	bool found = false;
	for(int i = 0; i < old_rows; i += 5) {
		if(markerID == state(i, 0)) {
			found = true;
		}
	}
	if(false == found) {
		return;
	}

	// Save current state and covariances
	Eigen::Matrix<double, Eigen::Dynamic, 1> state_tmp = state;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Pp_tmp = Pp;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Pm_tmp = Pm;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q_tmp = Q;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R_tmp = R;
	int new_rows = old_rows - 5;

	// Set states and covariances to 0
	state.setZero(new_rows, 1);
	Pp.setZero(new_rows, new_rows);
	Pm.setZero(new_rows, new_rows);
	Q.setZero(new_rows, new_rows);
	R.setZero(new_rows, new_rows);

	int i;
	int ii;
	// Copy values from the saved states/covariances to the new ones and skip the ones
	// which should be deleted
	for(i = 0, ii = 0; i < new_rows; i += 5, ii += 5) {
		if(markerID == state_tmp[i]){
			ii += 5;
		}
		state(i + 0, 0) = state_tmp(ii + 0, 0);
		state(i + 1, 0) = state_tmp(ii + 1, 0);
		state(i + 2, 0) = state_tmp(ii + 2, 0);
		state(i + 3, 0) = state_tmp(ii + 3, 0);
		state(i + 4, 0) = state_tmp(ii + 4, 0);
	}

	int j;
	int jj;
	for(i = 0, ii = 0; i < new_rows; i++, ii++) {
		if(0 == (i % 5)) {
			if(markerID == state_tmp[i]){
				ii += 5;
			}
		}
		for(j = 0, jj = 0; j < new_rows; j++, jj++) {
			if(0 == (j % 5)) {
				if(markerID == state_tmp[j]){
					jj += 5;
				}
			}
			Pp(i, j) = Pp_tmp(ii, jj);
			Pm(i, j) = Pm_tmp(ii, jj);
			Q(i, j) = Q_tmp(ii, jj);
			R(i, j) = R_tmp(ii, jj);
		}
	}

	/*
	ROS_INFO_STREAM("removeMarker: state: " << state.format(OctaveFmt));
	*/
}

/**
 * @brief Predicts the center of the marker in the next frame.
 */
void MarkerTracker::predict() {
	// Get current time and calculate time difference since last prediction
	std::chrono::system_clock::time_point time_now = std::chrono::system_clock::now();
	std::chrono::duration<double> time_diff = time_now - last_update;

	const int rows = state.rows();

	// Create system matrix A
	// (Identity matrix plus \delta_t for position ({x,y}_new = {x,y}_old + {v_x,v_y}_old * \delta_t)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
	A = Eigen::MatrixXd::Identity(rows, rows);
	for(int i = 0; i < rows; i += 5) {
		A(i + 1, i + 3) = time_diff.count();
	}

	//ROS_INFO_STREAM("A: \n" << A.format(OctaveFmt));

	// Predict new states
	state = A * state;

	// Predict new covarinces of the states
	Pp = A * Pm * A.transpose() + Q;

	/*
	ROS_INFO_STREAM("predict: state: " << state.format(OctaveFmt));
	ROS_INFO_STREAM("Pp: " << Pp.format(OctaveFmt));
	*/

	// Update time point
	last_update = time_now;
}

/**
 * @brief Get the predicted location of the markers.
 * @return Vector with the id and the location of the predicted markers.
 */
 std::vector<std::tuple<int, cv::Point2d>> MarkerTracker::getPredictedMarkers() {
  int rows_5 = state.rows() / 5;
  std::vector<std::tuple<int, cv::Point2d>> predictedPoints(rows_5);

	// Create vector of tuples containing marker id and position
	for(int i = 0; i < rows_5; i++) {
		std::get<0>(predictedPoints[i]) = state(i * 5 + 0, 0);
		std::get<1>(predictedPoints[i]).y = state(i * 5 + 1, 0);
		std::get<1>(predictedPoints[i]).x = state(i * 5 + 2, 0);
	}

	return(predictedPoints);
}

/**
 * @brief Updates the Kalman Filter (2nd step)
 * @param markers Observed markers (used as measurement)
 * @return Current state (markers).
 */
Eigen::Matrix<double, Eigen::Dynamic, 1> MarkerTracker::update(std::vector<aruco::Marker> markers) {

	// Check if detected markers are within the tracked markers.
	// Delete tracked markers which were not detected.
	bool found;
	for(int i = 0; i < state.rows(); i += 5) {
		found = false;
		for(const aruco::Marker& marker : markers) {
			if(0 > marker.id) {
				continue;
			}
			if(marker.id == static_cast<int>(state(i))) {
				found = true;
				continue;
			}
		}
		if(false == found) {
			removeMarker(state(i));
			i -= 5;
		}
	}

	// Return if state is empty
	if(0 == state.rows()) {
		return(state);
	}

	// Create measurement vector
	int rows = Pp.rows();
	Eigen::Matrix<double, Eigen::Dynamic, 1> measurements;// = state;
	measurements.setZero(rows, 1);

	// Set measurement values from the given markers and set velocity to 0
	// Temporary save markers which were detected but not yet tracked
	// to add them after the update step
	std::vector<aruco::Marker> markersToAdd;
	for(const aruco::Marker& marker : markers) {
		found = false;
		for(int i = 0; i < rows; i += 5) {
			if(marker.id == static_cast<int>(state(i))) {
				measurements(i + 0) = marker.id;
				measurements(i + 1) = marker.getCenter().y;
				measurements(i + 2) = marker.getCenter().x;
				measurements(i + 3) = 0;
				measurements(i + 4) = 0;
				found = true;
				continue;
			}
		}
		if(false == found) {
			markersToAdd.push_back(marker);
		}
	}
	rows = Pp.rows();

	//ROS_INFO_STREAM("update: measurements: " << measurements.format(OctaveFmt));
	// Create Kalman gain matrix (K)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
	K.resize(rows, rows);

	// Calculate PR (The inverse is later used)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PR_inv = Pp + R;
	/*
	ROS_INFO_STREAM("Pp: " << Pp.format(OctaveFmt));
	ROS_INFO_STREAM("R: " << R.format(OctaveFmt));
	ROS_INFO_STREAM("update: state: " << state.format(OctaveFmt));
	ROS_INFO_STREAM("PR: " << PR_inv.format(OctaveFmt));
	*/

	// Calculate the inverse
	PR_inv = PR_inv.inverse();
	//ROS_INFO_STREAM("PR_inv: " << PR_inv.format(OctaveFmt));

	K = Pp * PR_inv;

	//ROS_INFO_STREAM("K: " << K.format(OctaveFmt));

	// Create error vector
	Eigen::Matrix<double, Eigen::Dynamic, 1> error = measurements - state;
	// Update states with the Kalman gain matrix and the error
	state = state + K * error;

	// Update the covariances of the states with the Kalman gain matrix and the error
	Pm = (Eigen::MatrixXd::Identity(rows, rows) - K) * Pp;

	// Add detected markers which are not yet tracked
	if(false == markersToAdd.empty()) {
		addMarkers(markersToAdd);
	}

  // Update time point
  last_update = std::chrono::system_clock::now();

	/*
	ROS_INFO_STREAM("update: state: " << state.format(OctaveFmt));
	ROS_INFO_STREAM("Pm: " << Pm.format(OctaveFmt));
	*/

	return(state);
}
