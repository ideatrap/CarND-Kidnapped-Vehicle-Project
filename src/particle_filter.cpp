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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.

  // set the number of particles
	num_particles = 200;
	//resize weight
	weights.resize(num_particles);

	//resize particles
	particles.resize(num_particles);


	default_random_engine gen;
	//Normal distribution of x,y,yaw_rate

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  //Set standard deviations for x, y, and psi.
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

	// Initializes particles
	for (int i = 0; i < num_particles; i++) {
  // Add generated particle data to particles class
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0/num_particles;
				particles[i] = p;
        weights[i]=(p.weight);
  }

	is_initialized = true;
  return;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// Distributions for adding noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);


	double x, y, theta;
	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p = particles[i];

		if (fabs(yaw_rate) < 1e-4){
			// if yaw_rate is zero
			p.x = p.x + velocity  * cos(p.theta)* delta_t;
			p.y =  p.y + velocity  * sin(p.theta)* delta_t;
		} else{
			p.x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		}
        p.theta = yaw_rate * delta_t + p.theta;
		// add noise
		p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

		particles[i] = p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& observation:observations){
		double  best_distance = 9999999.0;
		for (const auto& p:predicted) {
			double distance = dist(p.x, p.y, observation.x, observation.y);
			if (distance < best_distance){
					observation.id = p.id;
					best_distance = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // All calculations will be performed in maps coordinate system,
  double sensor_range_sqrd = pow(sensor_range,2);

    for (int i = 0; i < num_particles; i++) {
			Particle &particle = particles[i]; // for each particle
			//First find all landmarks that are within range
      std::vector<LandmarkObs> inrange_landmarks;
			for (int j = 0; j < map_landmarks.landmark_list.size(); j++){ //for each landmark
            double distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particle.x, particle.y);
            if (distance <= sensor_range){ //valid landmark
								LandmarkObs landmark;
								landmark.id = j;
								landmark.x = map_landmarks.landmark_list[j].x_f;
								landmark.y = map_landmarks.landmark_list[j].y_f;

								inrange_landmarks.push_back(landmark);
            }
      }
      if (inrange_landmarks.size()>0){
        // now transform observations to map frame (to match inrange_landmarks coordinate system)
        // assuming the observations were made from the i'th particle's perspective
        std::vector<LandmarkObs> transformed_obs(observations.size());
				for (int j=0; j < observations.size(); ++j) {
            transformed_obs[j].x = observations[j].x * cos(particle.theta) - observations[j].y * sin(particle.theta) + particle.x;
            transformed_obs[j].y = observations[j].x * sin(particle.theta) + observations[j].y * cos(particle.theta) + particle.y;
        }
        //get associations
        dataAssociation(inrange_landmarks, transformed_obs);
        //update weights
        double w = 1.0;
				for (const auto observation:transformed_obs) {
								//cout << "use id: " <<observation.id  << endl;

		        int index = observation.id;
						double nn_x = map_landmarks.landmark_list[index].x_f;
		        double nn_y = map_landmarks.landmark_list[index].y_f;

						double x = observation.x;
		        double y = observation.y;

		        double std_x = std_landmark[0];
		        double std_y = std_landmark[1];

						//calculate multi-variate Gaussian distribution
		        double x_diff = (x - nn_x) * (x - nn_x) / (2 * std_x * std_x);
		        double y_diff = (y - nn_y) * (y - nn_y) / (2 * std_y * std_y);
		        w *= 1 / (2 * M_PI * std_x * std_y) * exp(-(x_diff + y_diff));
    		}
				particles[i].weight = w;
      }
      else{
        particles[i].weight = 0;
      }

    }
    //now normalize it
    double scale_factor = 0.0;
    for (int i = 0; i < num_particles; i++){
      scale_factor+=particles[i].weight;
    }
    // cout << scale_factor << endl;
    for(int i = 0; i < num_particles; i++){
      particles[i].weight = particles[i].weight/scale_factor;
      weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	random_device rd;
	default_random_engine gen(rd());
	// Vector for new particles
  vector<Particle> particles_new (num_particles);

	for (int i = 0; i < num_particles; ++i) {
    discrete_distribution<int> resample(weights.begin(), weights.end());
    particles_new[i] = particles[resample(gen)];

  }
  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
