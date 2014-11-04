# Create an RL Testbed
Want to be able to generate and experiment on many different types of (PO)MDPs.

I am thinking that, in contrast to the current implementation, everyhing should be set up in terms of matrices and operators, with the ability to generate runs in a reproducible way from the command line.

Output files should be YAML-style documents, recording the steps, the ideal values, and the configuration which generated all of the data

It would be smart to include things like the seed which generated all of the random data (for the various things making use of randomization) as well. Something like theano's randomstreams would be good here.

# Create a feature vector generating library