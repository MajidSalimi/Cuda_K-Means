
#ifndef POWER_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define POWER_H


int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
	int      nfeatures,				/* number of attributes for each point */
	int      npoints,				/* number of data points */
	int      nclusters,				/* number of clusters */
	int     *membership,				/* which cluster the point belongs to */
	float  **clusters,				/* coordinates of cluster centers */
	int     *new_centers_len,		/* number of elements in each cluster */
	float  **new_centers				/* sum of elements in each cluster */
	);


#endif