/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Check, if already initialized
	if (is_initialized) {
		return;
	}

	// Initialize number of particles
	num_particles = 100;

	// Extract standard deviations for x, y, and theta
	const double std_x = std[0];
	const double std_y = std[1];
	const double std_theta = std[2];

	// Create normal (Gaussian) distribution for x, y, and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Initialize particles
	for (int i=0; i<num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		// Add to particles vector
		particles.push_back(particle);
	}

	// The filter is now initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Extract standard deviations for x, y, and theta
	const double std_x = std_pos[0];
	const double std_y = std_pos[1];
	const double std_theta = std_pos[2];

	// Create normal (Gaussian) distribution for x, y, and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Predict update
	for (int i=0; i<num_particles; i++) {  // For each particle
		const double theta = particles[i].theta;

		if (fabs(yaw_rate) < EPS) {
			// yaw is not changing
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add random Gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Extract vector sizes for performance purposes
	const int numObservations = observations.size();
	const int numPredictions = predicted.size();

	for (int i=0; i<numObservations; i++) {  // For each observation
		// Initialize min distance as a really big number
		double minDistance = numeric_limits<double>::max();

		// Initialize the found map in something not possible.
    	int mapId = -1;

		for (int j=0; j<numPredictions; j++) {  // For each predition
			// Calculate distance
			const double xDistance = observations[i].x - predicted[j].x;
			const double yDistance = observations[i].y - predicted[j].y;
			// NOTE: Since we just compare the distances, we can skip the sqrt()
			const double distance = xDistance * xDistance + yDistance * yDistance;

			// If the "distance" is less than minima, store the id and update minima
      		if (distance < minDistance) {
        		mapId = predicted[j].id;
        		minDistance = distance;
      		}
		}

		// Update the observation identifier.
    	observations[i].id = mapId;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Extract landmark measurement uncertainties
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];

	// Power of sensor range, used for distance comparison
	const double sensor_range_2 = sensor_range * sensor_range;

	// Constants used for calculating the weights
	const double term_exp_x = .5 / (std_x * std_x);
	const double term_exp_y = .5 / (std_y * std_y);
	const double term_base = 2.0 * M_PI * std_x * std_y;

	// Extract vector sizes for performance purposes
	const int numLandmarks = map_landmarks.landmark_list.size();
	const int numObservations = observations.size();

	for (int i=0; i<num_particles; i++) {  // For each particle
		const double pX = particles[i].x;
    	const double pY = particles[i].y;
    	const double pTheta = particles[i].theta;

		/**********************************************************************
		 * STEP 1: Find landmarks within sensor range
		 *********************************************************************/
		vector<LandmarkObs> landmarks_in_range;
		for (int j=0; j<numLandmarks; j++) {  // For each landmark
			// Extract landmark data
			const int id = map_landmarks.landmark_list[j].id_i;
			const float landmarkX = map_landmarks.landmark_list[j].x_f;
			const float landmarkY = map_landmarks.landmark_list[j].y_f;
			
			// Calculate distance
			const double xDistance = pX - landmarkX;
			const double yDistance = pY - landmarkY;
			// NOTE: Since we just compare the distances, we can skip the sqrt()
			const double distance = xDistance * xDistance + yDistance * yDistance;

			// If the "distance" is less than "sensor range", store the landmark
			if (distance < sensor_range_2) {
				landmarks_in_range.push_back(LandmarkObs { id, landmarkX, landmarkY });
			}
		}

		/**********************************************************************
		 * STEP 2: Transform observation coordinates to map coordinates
		 *********************************************************************/
		vector<LandmarkObs> map_observations;
		for (int j=0; j<numObservations; j++) {  // For each observation
			// Extract observation data
			const int oId = observations[j].id;
			const double oX = observations[j].x;
			const double oY = observations[j].y;

			// Transform coordinates
			const double mX = pX + cos(pTheta) * oX - sin(pTheta) * oY;
			const double mY = pY + sin(pTheta) * oX + cos(pTheta) * oY;
			map_observations.push_back(LandmarkObs { oId, mX, mY });
		}

		/**********************************************************************
		 * STEP 3: Associate landmarks in range to landmark observations
		 *********************************************************************/
		dataAssociation(landmarks_in_range, map_observations);

		/**********************************************************************
		 * STEP 4: Update particle weights
		 *********************************************************************/
		const int numMapObservations = map_observations.size();
		const int numLandmarksInRange = landmarks_in_range.size();
		double weight = 1.0;
		for (int j=0; j<numMapObservations; j++) {  // For each mapped observation
			// Extract observation data
			const int oId = map_observations[j].id;
			const double oX = map_observations[j].x;
			const double oY = map_observations[j].y;

			// Get associated landmark positions
			double lX, lY;
			for (int k=0; k<numLandmarksInRange; k++) {  // For each landmark in range
				if (landmarks_in_range[k].id == oId) {
					lX = landmarks_in_range[k].x;
					lY = landmarks_in_range[k].y;
					break;
				}
			}

			// Calculate distance
			const double dX = oX - lX;
			const double dY = oY - lY;

			// Calculate weight
			const double p = exp(-term_exp_x * dX * dX - term_exp_y * dY * dY) / term_base;
			if (p < EPS) {
				weight *= EPS;
			}
			else {
				weight *= p;
			}
		}
		particles[i].weight = weight;
		weights[i] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
