# capsules
Implementation of capsule network with dynamic routing that actually works.    

New features:
1. modification of the loss for anomaly detection based on https://arxiv.org/pdf/1909.02755.pdf    
2. anomaly score based on https://arxiv.org/pdf/1909.02755.pdf   
3. normality scores based on https://arxiv.org/pdf/1907.06312.pdf   
4. HitOrMiss capsules and Centripetal Loss from https://arxiv.org/pdf/1806.06519.pdf   
5. Kernelized Capsule Networks (https://arxiv.org/abs/1906.03164) example with GPytorch - partially done, reconstruction doesn't work yet   

TODO:
1. add CIFAR10 example
2. train CatsVsDogs to above 80% validation accuracy
3. add Autoencoder example
4. add unsupervised capsule layer
