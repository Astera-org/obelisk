package main

/*
Types of parameters we can search over:
 float range
 int range
 set of values
 gaussian search around a particular point (seems like our search algo should just realize this)
 step size in range

 Functions:
	load parameters from .cfg file
	Get the next point to search
	Load search from DB
	Parse job to determine what point was searched

*/

type ParamRange struct {
	name     string
	max      float64
	min      float64
	stepSize float64
}

type Search struct {
	searchID   int32
	dimensions []ParamRange
}
