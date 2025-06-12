# PseudoDiloco

My experiments with [DiLoCo](https://arxiv.org/pdf/2311.08105). I don't have multiple machines so I'm simulating multiple clients/replicas on my one local GPU (hence the "Pseudo").

So far working with ResNets and CIFAR-10, and DiLoCo has proved effective here on IID data, getting to > 90% validation accuracy on multiple different communication frequencies and number of workers. 

Non-IID data is in progress: the plan is to embed the training data, perform K-means (where K is the number of replicas), then distribute data accordingly. I would expect to see worse performance here,  after which I'll try applying some previous Federated Learning techniques to DiLoCo (e.g. FedProx).
