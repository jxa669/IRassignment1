from geometry_msgs.msg import Pose, PoseArray, Quaternion
from pf_base import PFLocaliserBase
import math
import rospy
import numpy

from util import rotateQuaternion, getHeading
import random

from time import time
from sensor_msgs.msg import LaserScan

class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
 	self.ODOM_ROTATION_NOISE = 0.1
	self.ODOM_TRANSLATION_NOISE = 0.1
	self.ODOM_DRIFT_NOISE = 0.1

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 200     # Number of readings to predict
       
	self.scan = LaserScan
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
	self.particlecloud = PoseArray()
	i = 0
	while i < self.NUMBER_PREDICTED_READINGS:
		temppose = Pose()
		temppose.position.x = initialpose.pose.pose.position.x
        	temppose.position.y = initialpose.pose.pose.position.y
        	temppose.position.z = initialpose.pose.pose.position.z
		temppose.orientation = initialpose.pose.pose.orientation
		self.particlecloud.poses.append(temppose)
		i+=1
	for p in self.particlecloud.poses:
		p.position.x += self.ODOM_TRANSLATION_NOISE*random.gauss(0,5)
		p.position.y += self.ODOM_DRIFT_NOISE*random.gauss(0,5)
		p.orientation = rotateQuaternion(p.orientation, self.ODOM_ROTATION_NOISE*random.gauss(0,2))
	return self.particlecloud

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
	self.scan = scan
	likelihood =  []
	for p in self.particlecloud.poses:
		likelihood.append(self.sensor_model.get_weight(scan,p))
	likelihoodnew = []
	for x in likelihood:
		likelihoodnew.append(x/sum(likelihood))
	CDF = numpy.cumsum(likelihoodnew)
	u = (1-random.random())/len(CDF)
	i=0
	j=0
	particlecloudnew = PoseArray()
	while j < len(CDF):
		while u > CDF[i]:
			i=i+1
		particlecloudnew.poses.append(self.particlecloud.poses[i])
		u = u + 1/float(len(CDF))
		j = j + 1
	self.particlecloud.poses = particlecloudnew.poses
	particlecloudnew = PoseArray()
	for p in self.particlecloud.poses:
		pose = Pose()
		pose.position.x = p.position.x + self.ODOM_TRANSLATION_NOISE*random.gauss(0,2)
		pose.position.y = p.position.y + self.ODOM_DRIFT_NOISE*random.gauss(0,2)
		pose.orientation = rotateQuaternion(p.orientation, self.ODOM_ROTATION_NOISE*random.gauss(0,2))
		particlecloudnew.poses.append(pose)
	self.particlecloud.poses = particlecloudnew.poses	
	print(self.particlecloud)

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
	estimatedpose = self.particlecloud.poses[0]
	for p in self.particlecloud.poses:
		if self.sensor_model.get_weight(self.scan,p) > self.sensor_model.get_weight(self.scan,estimatedpose):
			estimatedpose = p
	return estimatedpose
