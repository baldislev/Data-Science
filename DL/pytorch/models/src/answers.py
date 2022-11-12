# ==============
# Part 1 answers

part1_q1 = r"""
1 - True.
By the definition, the in-sample-error is the error that we get upon evaluating on the model over the test set.

2 - False.
The ratio of the split provides different results. The bigger the test set the more likely for the model to overfit,
while small test set will probably lead to underfitting scenario.

3 - False.
Cross validation uses different splits of the test set to simulate validation process.

4 - True.
This is the whole purpose of CV process, to simulate a validation evaluation using different splits of the test set.

"""

part1_q2 = r"""
The approach is justified although might not always work well due to the imbalanced split, in that case CV 
method for tuning hyperparameters would be preferred.
Iw we assume that the split is balanced, then the proposed approach is similar to train, validation, test
split method for hyperparameters tuning with the test acting as validation set and absence of actual test 
set for final evalutaion.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
As we can infer from the graph, there is a decreasing order of connection between Accuracy and k parameter.
In other words, increasing k leads to lower Accuracy of the model's cv results. Based on that we can conclude
that increasing k does not lead to improved generalization for unseen data in this specific case.
It can be explained by the knn predict algorithm. Small k means making decisions based on the closest 
datapoints in the set, while big k means that even further samples impact the prediction.
For extreme values we get that for k=1 that the prediction will be as the label of the closest datapoint 
in the trainset, and for k=N - prediction is the majority label of the whole trainset.
But in general case it is very possible to see a situation in which the k that generalizes the most is 
somewhere between 1 and N.
"""

part2_q2 = r"""
1. While choosing the best model with respect to train-set accuracy we might easily overfit, since 
there is no indication on generalization in such method of tuning. CV, on the other hand will use 
iterative splits of the trainset as validation set and so we don't rely solely on trainset performance when
choosing the best model.

2. In case the train-test split is imbalanced, such process will yield a biased model that is fitted 
well on the test set, but with poor generalization. 
The iterative method of CV with different validation subsets in each iteration prevents this problem from
happening.  
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The delta is arbitrary to allow us to define for ourselves the margin in which we want to
penalize over the wrong predictions of the samples.
"""

part3_q2 = r"""
1. Based on the visualization of the weight vector as an image, we can conclude that the model learns 
a filter for each class. This filter seems to abstractly generalize the shape of an image that belongs to 
the relevant class based on all of the samples of that class. For example the filter for class '0' has 
high values at places where we expect to have pixels that belong to 0, and low values otherwise.
The classification errors might be explained by the fact that some digits are very similar to each other
(e.g. 4 and 7), and their images might deviate in such way that they will fit better for filter of the other class
rather than to image's own class.

2. The difference lies in the way both models make predictions:
KNN will predict the majority label of k train samples that are l2 closest to the input, while SVM will
apply filters that are affected by all samples in the trainset and predict the class which filter 
correlates the most with the shape of an input image.
"""

part3_q3 = r"""
1. The chosen learning rate is somewhat High, we can see it since between epochs 10 and 15
we have a significant jump of the loss function which can be explained by a high learning rate
which oversteps the local minimum.
For the Good learning rate we expect to see a smooth declining curve that converges exactly at the last epoch.
For Too Low learning rate we wouldn't see a convergence in the given number of epochs.

2. Based on the accuracy graph, we can infer that the model is slightly overfitted to the training set.
It can be explained by the train accuracy being first of all around 95% and the test accuracy is 
simply slightly lower than that. 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern would be that the difference between y and $\hat{y}$ is close to zero for all $\hat{y}$.

According to the plots, the trained model gives good results compared to the top-5 features model, Most of the points 
inside the "epsilon sleeve" around zero as we wanted.

"""

part4_q2 = r"""
1. Yes it's still a linear model, we did not change the hypothesis class. 

2. Yes we can take any nonlinear function and fit the original features with that function. In case this function
can be computed.

3. Adding non-linear features affect the decision boundary of such a classifier by adding more dimensions to it, It is
 still a hyperplane just from another dimension. A good selection of non-linear features can transform the problem
 to be linearly separable.
"""

part4_q3 = r"""
1. Because we want to sample more values that are close to 0 than 100, There is a higher chance that the optimal
 $\lambda$ will be within this value range.
2. There where 2 values off `degree_range` and 20 values of `lambda_range` therefore total of 40 times.
"""
# ==============
# Part 5 (Backprop) answers

part5_q1 = r"""
**1.** <br>
**A:** Y is of dimensions [64, 512], X is [64, 1024] where $Y = \mat{X} \cdot \mattr{W}$ <br>
therefore $\pderiv{\mat{Y}}{\mat{X}}$ has dimensions of [64, 512, 64, 1024].<br>
<br>
**B:** It is indeed sparse. Each tensor at index [i, j] is a matrix [64, 1024] which represents <br>
partial derivative of Y[i, j] with respect to X. <br>
Y[i, j] = X[i, : ] * W[j, :], which means it depends only on i'th row of X. <br>
Therefore $\pderiv{\mat{Y}}{\mat{X}}$[i, j] which is a [64, 1024] tensor, will have zeroes everywhere, <br>
except for i'th row, which will be populated with partial derivative values - j'th row of W. <br>
<br>
**C:** We don't have to calculate the $\pderiv{\mat{Y}}{\mat{X}}$ explicitly in order to produce $\delta\mat{X}$, <br>
since $\delta\mat{X}=\delta\mat{Y}\cdot\pderiv{Y}{\mat{X}}$ so we only need the product itself. <br>
As we mentioned above, $\pderiv{Y}{\mat{X}}$ is sparse and we will use this feature for the calculation. <br>
$\delta\mat{X}$ has dimensions of [64, 1024], and $\delta\mat{X} = \sum_{i,j}\delta\mat{Y}[i, j]\cdot\pderiv{\mat{Y}}{\mat{X}}[i, j]$. <br>
As we noticed above, $\pderiv{\mat{Y}}{\mat{X}}[i, j]$ is a sparse matrix where i'th row is j'th row of W. <br>
Therefore, $\delta\mat{X} = \sum_{j}\delta\mattr{Y}_{col_j}\cdot\mat{W}_{row_j} = \delta\mat{Y}\cdot\mat{W}$ <br>
<br>
**2.** <br>
**A:** Y is of dimensions [64, 512], W is [512, 1024] where $Y = \mat{X} \cdot \mattr{W}$ <br>
therefore $\pderiv{\mat{Y}}{\mat{W}}$ has dimensions of [64, 512, 512, 1024].<br>
<br>
**B:** It is indeed sparse. Each tensor at index [i, j] is a matrix [512, 1024] which represents <br>
partial derivative of Y[i, j] with respect to W. <br>
Y[i, j] = X[i, : ] * W[j, :], which means it depends only on j'th row of W. <br>
Therefore $\pderiv{\mat{Y}}{\mat{W}}$[i, j] which is a [512, 1024] tensor, will have zeroes everywhere, <br>
except for j'th row, which will be populated with partial derivative values - i'th row of X. <br>
<br> 
**C:** We don't have to calculate the $\pderiv{\mat{Y}}{\mat{W}}$ explicitly in order to produce $\delta\mat{W}$, <br>
since $\delta\mat{W}=\delta\mat{Y}\cdot\pderiv{Y}{\mat{W}}$ so we only need the product itself. <br>
As we mentioned above, $\pderiv{Y}{\mat{W}}$ is sparse and we will use this feature for the calculation. <br>
$\delta\mat{W}$ has dimensions of [512, 1024], and $\delta\mat{W} = \sum_{i,j}\delta\mat{Y}[i, j]\cdot\pderiv{\mat{Y}}{\mat{W}}[i, j]$. <br>
As we noticed above, $\pderiv{\mat{Y}}{\mat{W}}[i, j]$ is a sparse matrix where j'th row is i'th row of X. <br>
Therefore, $\delta\mat{W} = \sum_{j}\delta\mattr{Y}_{row_j}\cdot\mat{X}_{row_j} = \delta\mattr{Y}\cdot\mat{X}$ <br>
<br>
"""

part5_q2 = r"""
Backpropagation is a useful technique for differentiating parameters of the model by automatic building of a computational <br>
graph, which saves a lot of work for the developer by backpropagating the gradients through the graph to easily write and <br>
maintain neural network's code. But it is not required since the developer can manually implement the gradient calculations for each step of the model. <br>
It can be very painful or even unfeasible in some cases but theoretically possible.
"""


# ==============
# Part 6 (Optimization) answers


def part6_overfit_hp():
    wstd, lr, reg = 0.2, 0.02, 0.05
    return dict(wstd=wstd, lr=lr, reg=reg)


def part6_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.3,
        0.035,
        0.0029411022161431,
        0.00014853239474271534,
        0.005,
    )
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part6_dropout_hp():
    wstd, lr, = (
        0.2,
        0.0016,
    )
    return dict(wstd=wstd, lr=lr)



part6_q1 = r"""
**No-dropout** has the lowest train loss and highest train accuracy with the logically opposite test results - <br>
as it was expected for a classic overfitting scenario. <br>
**Highest dropout** value (0.8) resulted in the worst train loss and train accuracy - it is quite understandable since we <br>
tailored the model's hp specifically to overfit for zero dropout and so the maximum dropout was expected to perform <br>
the worst on the train set. At the same time, test loss and accuracy are better here. <br>
Finally, the **moderate dropout** (0.4) performed the best on the test dataset. Again, an expected result since zero dropout
was tailored to be the worst, maximum dropout was an extreme value from another side and so it was expected for an intermediate <br>
value to perform better. <br>
**To sum up:** <br>
On train datasets both loss and accuracies performed better according to the order of the dropout value: the lower the better. <br>
Again, this is a predictable result of choosing hp that overfit on the zero dropout value.
On test data the intermediate dropout performed the best and the zero - worst. 
"""

part6_q2 = r"""
It is possible, since cross entropy loss compares probabilities of classes and therefore not directly connected <br>
to the accuracy which is based on predictions that use those probabilities. <br>
It is possible that during an epoch some wrongly classified samples already had probabilities that are very close <br>
to the threshold value and so a slight improvement of their scores can provide a better accuracy. To counterpart this <br>
improvement of the loss, some other samples probabilities can become much less similar to the real distribution of classes <br>
and so overall the general loss will be increased.
"""

part6_q3 = r"""
1. **Gradient descent** is an iterative optimization process which aims to find the minimum of the loss function by updating <br>
each parameter of the function by 'stepping' in the opposite direction of the gradient. <br>
**Backpropagation**, on the other hand is a tool for calculating the partial derivatives of the loss function with respect <br>
to it's parameters by using a chain rule of differentiation. <br>
<br>
2. **GD** is a descent method that uses all of the training samples for the loss and gradient calculations at each step.
**SGD** on the other hand at each step will calculate the loss and the gradient based on one or a few randomly chosen samples from <br>
the training set. From one point such approach can result in a poor approximation of the real gradient, <br>
which means we won't go exactly where we need to, the time for convergence increases or we even might get lost. <br>
But from another point there is a mathematical justification for SGD which says that on average SGD converges to GD. <br>
It can be shown with some basic probability tools. More than that, in some cases, stochastic approach can help to escape unwanted 
poor local minimum of the loss function.<br>
<br>
3. SGD is more often used simply because classic GD computationally is significantly demanding and sometimes even infeasible to compute. <br>
Training data can be unbounded and it is very possible that there won't be enough memory to allocate for such a computation. <br>
SGD doesn't have such problems, the ability to practically compute it, even on machines that don't possess an impressive <br>
amount of memory makes it's usage very popular. <br>
<br>
4. <br>
A: Yes, this approach will provide a gradient equivalent to that of GD. The final loss tensor will hold the computational <br>
graph which is logically equal to that of GD case. And so traversing both graphs and updating the parameters with respect to their <br>
gradients will produce the same result. <br>
B: The computational graph of final loss tensor will be as wide as the number of batches and so traversing it can lead to <br>
out of memory situation since train data is unbounded and therefore the width of a graph. <br>
"""


# ==============


# ==============
# Part 7 (MLP) answers


def part7_arch_hp():
    n_layers = 3  # number of layers (not including output)
    hidden_dims = 4  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part7_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.02, 0.013, 0.65  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part7_q1 = r"""
1. As we can see, our model converged on high values for both test and train accuracies, but arguably, we could<br>
have done better. The decision boundary plot shows an area with significant amount of misclassified red dots. <br>
Therefore there is definitively a degree of optimization error that can be reduced by increasing the optimization time <br>
or in other words - by training our model for more epochs. <br>

2. Judging the train and test loss graphs it seems that there is a small Generalization error. We can see that test
losses are as high as train losses and at some areas even have a lower value. The accuracies graphs show that test
accuracy is not that much lower than the train accuracy and sometimes even higher.

3. Approximation error is exposed by high test loss/accuracy or unsuitable decision boundary, meaning that the model is
not rich enough to deal with the characteristics of the data distribution. But as we can see we on the graphs and desicion <br>
boundary plot, our model is quite capable to deal with the data, therefore the approximation error is not high.
"""

part7_q2 = r"""
We would expect higher FNR. As we can see from the data generation process plots there is a dense area of negative <br>
samples in the train set while the same area in the validation set has more positive samples. <br>
Therefore our model will learn to predict negative class for this area of the validation set and those will be false negatves. <br>
As we can see from the confusion matrix, indeed there is a higher FNR than FPR.
"""

part7_q3 = r"""
1. In this case high FNR only means that the person will develop non-lethal symptoms that immediately will <br>
provide a diagnosis and the right treatment. On the other hand high FPR will significantly increase the cost. <br>
And so we will seek a model with lowest FPR and won't mind high FNR. A suitable point on roc curve would be one that is <br>
very close to the left but still high enough since we don't want FNR=1. <br>

2. In this case, high FNR is on the opposite dangerous since it means loosing a life of a patient with high probability. <br>
High FPR at the same time increases the cost of diagnosis process and imposes risk to a patient with no need. <br>
Therefore we would like to have equally low FPR and FNR and the point closest to (0,1) on the roc curve would be the optimal.
"""


part7_q4 = r"""
<br>
**When we write 'performance' here, we mean the combination of test_acc and valid_acc.** <br>
1. For fixed depth = {1, 2} it seems that there is a similar behaviour when changing the depth: <br>
the higher the width the better the model's performance and decision boundaries seem to be more fit to the data distribution.<br>
Whereas for depth = 4, the performance seem to fluctuate over the increasing width as well as the shape of the desicion boundaries <br>
seem to change rapidly. The best test performance and decision boundary was actually achieved for the smallest width. <br>
<br>
2. For fixed width = 2, as we increase the depth we get better performance and more adjust shape of the decision boundaries <br>
For width = 8, increasing the depth from 1 to 2 brings boost of performance and better boundaries, but the change of depth to 4 <br>
produces even lower performance and worse boundaries to the initial ones. <br>
For width = 32, it seems that increasing the depth doesn't make any significant change, but the overall performance slightly increased. <br>
For width = 128, the change of depth from 1 to 2 brings out a slight boost of performance and the shape of the boundaries. <br>
But increasing the depth to 4 drastically worsens the overall performance and boundaries. <br>
<br>
3. Model with d=1, w=32 has much better test and validation accuracies as well as decision boundaries than the model with d=4, w=8. <br>
While model with d=1, w=128 has worse performance and decision boundaries rather the model with d=4, w=32. <br>
Although for both cases the models have the same number of parameters, those are not equal, since depth brings exponential growth of <br>
expressivity of the model compared to the width's influence. <br>
Therefore in the first case we can infer that the model with higher depth overfitted, while in the second case, richer <br>
expressivity helped to perform better. <br>
<br>
4. It seems like in all cases, except from the two exceptions of w=32 and d={1,2} where the threshold selection resulted in <br>
only a slight decrease of test performance, the threshold selection much improved the test accuracies. <br>
It can be explained by the way this selection works: a threshold whose (FPR, TPR) are the closest to the point (0, 1) was chosen. <br>
Which means we choose a threshold that maximizes both TPR and TNR and as a result we get less mistakes and a better performance. <br>
"""

# ==============
# Part 8 (CNN) answers


def part8_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.functional.cross_entropy  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.1, 0.02, 0.05  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part8_q1 = r"""
1. Following the notation from the tutorial, The number of parameters per convolutional layer is $(F^2*Cin+1)*Cout$
 Hence for the regular block we get $(3^2*256+1)*256 + (3^2*256+1)*256 = 1,180,160$
 and for the bottleneck block we get $(1^2*256+1)*64 + (3^2*64+1)*64 + (1^2*64+1)*256= 70,016$
2. From the previous section we can conclude that there is more floating point operations in the regular case (compare
 to bottleneck), This is because operations are performed at higher dimensions comparing the params $(F, Cin, Cout)$
  in each case.
3. In the regular case we have the ability to combine the input within feature maps and across feature maps (as we saw
 in the tutorial). In the bottleneck case in the layers (1x1  64) we combine only across feature maps because of the
 kernel size 1x1. 
"""

# ==============
# Part 9 (CNN Experiments) answers


part9_q1 = r"""
1. We got the best results for depth 2, probably because of vanishing gradients problem. We can see from the graph that
 even for L4 we stop the learning relatively (to L2) early.
2. The network wasn't trainable for L8, L16, probably caused by vanishing gradients.
 We can try to solve this problem by using Resnet architecture as we learned in the tutorial or we can try different
 activation function.
"""

part9_q2 = r"""
As we could expect from experiment 1.1 the network learn only for L2 and L4 (not for L8).
For L2 we got the best accuracy and for K256, It's seems like we get better result and learn faster for higher number
of filters. For L4 we got similar results just with less accuracy than L2 (as we could expect from experiment 1.1).
"""

part9_q3 = r"""
As we concluded in the previous experiment it seems that for higher number of filters, and for small number of 
layers the network learn faster and we get better accuracy. But unlike previous experiments now the network learned 
only for L1 and L2 and not for L3 and L4.
"""

part9_q4 = r"""
In the same way as we have seen in previous experiments the network learn only for L2 K64_128_256. For higher values
of L the network simply did not learn.
"""

part9_q5 = r"""
1. In the YourCNN class we used Resnet architecture with a skip connection between every two convulsions. 
 This allowed us to train deeper networks and maintain a relatively large number of filters. In addition, we performed
 a droupout and batchnorm which significantly improved the accuracy of the test.
2. In the second experiment we were able to train significantly larger networks, and as a result we obtained higher
 accuracy on both the training group and the test group (an accuracy of about 85% compared to the first experiment in
 which we received a maximum of 60%).
 """

 # ==============
# Part 10 answers


def part10_rnn_hyperparams():
    hypers = dict(
        batch_size=256,
        seq_len=64,
        h_dim=512,
        n_layers=3,
        dropout=0.4,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=4,
    )

    return hypers


def part10_generation_params():
    start_seq = "ACT I. SCENE 1.\n" \
                "Rousillon. The COUNT'S palace\n\n" \
                "Enter BERTRAM, the COUNTESS OF ROUSILLON, HELENA, and LAFEU, all in black"
    temperature = 0.3
    return start_seq, temperature


part10_q1 = r"""
We split the corpus into sequences instead of training on the whole text because we want to parallelize the learning 
proses. Meaning we want to work with batches so we can utilize our hardware. also, the length of the text implies
directly on the ability to train, the longer the sequence the model gets more complicated in terms of calculating 
gradient. Moreover, we don't want to work with large objects and we don't need to, this is because that the importance
of character for the next prediction is less mining full if it's furder away (relative to the predicted character).
"""

part10_q2 = r"""
When we generate a character we use not only the information we get in the start sequence but also the characters we
generated up until the current prediction. That is, in each iteration, the predicted character is chained to the 
generated text.
"""

part10_q3 = r"""
Because we want to preserve the semantic value of the text. If we will shuffle the order of batches we will actually
try to learn a permutation of the text.
"""

part10_q4 = r"""
1. We lower the temperature for sampling because we want the character with the highest score to have a higher
probability of being sampled relative to the others. As explained a low $T$ will result in less uniform distributions.
2. When the temperature is very high we get almost uniform distributions because of the high variance, Which leads to
a random selection of characters and content less similar to the start sequence.
3. When the temperature is very low we get the opposite, low variance which means that the generated text will be as
close as it's can to the original text (assuming the model is well trained).
"""


# ==============
# Part 11 answers

PART11_CUSTOM_DATA_URL = None


def part11_vae_hyperparams():
    hypers = dict(
        batch_size=128, h_dim=256, z_dim=16, x_sigma2=0.001, learn_rate=5e-5, betas=(0.9, 0.99),
    )

    return hypers


part11_q1 = r"""
One of the main usages of $\sigma^2$ is to give control over the regularization. More specifically, it delegates the importance of the
data reconstruction over the KL divergence. And thus we can control how precisely we want to model the training data.
The higher the value of $\sigma^2$ the the stronger the regularization is and thus the training concentrates more on the generalization
of the distribution and not on the precision of the train data reconstruction.
Another way to look at $\sigma^2$ is as a way of telling how big is uncertainty in the generational process, since 
$p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z})$.
Selecting high $sigma^2$ will produce a more wider distribution which means after training is complete, the latent space will be more spread
and continuous which means it will be easier to generate unseen fake images. But at the same time similar images might be quite distant in the latent space.
Low $sigma^2$ might lead us to a very dense latent space in which there is overlapping of features and thus generating meaningful samples
might get difficult.
"""

part11_q2 = r"""
1.
The data_loss term describes the distance between the original data and the reconstructed data from the decoder. Minimizing this term
will lead to better approximation of the original data by the latent space.

kldiv_loss term of the loss shows how much the modeled posterior distribution $q_\alpha$ is different from the standard Gaussian distribution. 

2.
KL loss term pushes the latent-space distribution to be more similar to the Standard Gaussian Distribution, as we saw in the lecture this is a
regularization term, it guides encoder to create a better, continuous encodings such that the decoder will train on them. 

3. 
The benefit of this effect is that we can use the $/sigma^2$ hyperparameter for a tradeoff between real data reconstruction precision and 
generalization of the problem. Also as we stated above, this term pushes the latent space to acquire relevant features such as continuity and
meaningfulness for the decoder.
"""

part11_q3 = r"""
In this task learn to generate data by training on samples that 'live' in a very high dimensional space (images) but we interested only in a specific 
subspace (G Bush images) that we believe has much lower dimension and so we model a smaller latent space that hopefully 
will represent the evidence space (training samples). Naturally our goal is to maximize the probability of the 
training set (evidence distribution) which will be described in the encoder/decoder language. 
As we saw in the lecture it is computationally infeasible, therefore instead we try to model the lower bound of P(evidence) by 
modelling posterior distribution as $q_\alpha$ and pushing it to be as closer as possible to the real posterior distribution.
"""

part11_q4 = r"""
The main reason of modeling the $log\sigma^2$ instead of $\sigma^2$ is to ensure better numerical stability. 
Log transformation of variance which has domain of [0, inf) and usually has values that are close to zero will allow to model those
close to zero values with a better numerical representation and thus such technique is preferred when dealing with the vanishing gradients 
case that can easily happen when modeling the $\sigma^2$ directly.
"""

# ==============
# Part 12 answers

PART3_CUSTOM_DATA_URL = None


def part12_gan_hyperparams():
    hypers = dict(
        batch_size=32,
        z_dim=8,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=1e-4,
            betas=(0.5, 0.99),
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2.5e-3,
            betas=(0.6, 0.99),
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part12_q1 = r"""
GAN consists of two independent but connected to each other parts: the Discriminator and the Generator.
The main idea is that the Generator samples data such that it will fool the Discriminator, but at the same time there is a
competitive dual objective in which the Discriminator gets fake (from Generator) and real images and attempts to make the right classifications.
Each batch is used to deal with the both objectives.
When the Generator is trained, we sample data with gradient tracking in order to optimize the "fooling" action on the Discriminator by updating
the Generator's weights.
When the Discriminator is trained, we are not interested in training the Generator, but we do need the fake data the Generator is producing.
So we sample it without the gradient tracking as if this data came from somewhere as the real data did. We use both real and fake images
to train the Discriminator's weights to make the right classifications.
"""

part12_q2 = r"""
1. No, we shouldn't stop the training based only on the Generator's low loss. In case the Discriminator has a high loss, it means 
that it's easy to fool it then our Generator is not really doing a good job. The whole concept of GAN is for Discriminator to be able to
make a good classification but for the Generator to still be able to fool it. So both losses should be low.

2. If the Generator's loss decreases it means it gets better and better at fooling the Discriminator.
While the Discriminator's loss constant values means it cancels the Generator's improvements but probably not too much to converge on a
desired result. For the purpose of learning we would expect both losses to decrease so that at the same time Discriminator is getting better
at differentiating fake and real data as the Generating is improving it's ability to fool the Discriminator.
"""

part12_q3 = r"""
The main difference between them is the sharpness of the image and the background behind the figure.
In VAE we generated blurry images with the simpler vague backgrounds because we are trying to minimize the MSE with 
respect to the original images. As a result, we got images that were similar to each other, and actually similar to the 
average image in the original data set.
In GAN, we generated sharper images with higher diversity with respect to the background. We minimize the loss with 
respect to the generated images and get images closer to the real distribution.
"""

# ==============