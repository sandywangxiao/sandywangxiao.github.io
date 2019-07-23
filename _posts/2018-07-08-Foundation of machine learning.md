---
layout: mypost
title: Foundation of machine learning
categories: [Machine Learning]
---

# 1.Introduction

+ **spaces:**   input space $\mathcal{X} $(vectors of explanatory variables) ; output space $\mathcal{Y} $
+ **Loss function:**  measures the cost of predicting y using estimator 
    + Square error loss:$ L(y,\hat{y})=(y-\hat{y})^2$
    + Hinge loss: $L(y,\hat{y})=max(0,1-y\hat{y})=|1-y\hat{y}|_+ $
    + Cross entropy loss:  $L(y,\hat{y})=-yln(\hat{y})-(1-y)ln(1-\hat{y}) $
           
+ **Types of Learning:**
![1](1.png)
+ **Three critical terms**
    + **Convexity:** convex problems can be solved efficiently $\Rightarrow$ Convexification
    + **Sparsity: ** common in high dimensional or massive data sets
    + **Assumptions: ** statistical guarantees
+ **Generalization/prediction error:**  $ R(f)=\mathbb{E}_{(x,y)\sim \mathcal{D}}L\{f(x),y\}=\mathbb{E}_x\mathbb{E}_{y|x}L\{f(x),y\} $
<br>**Empirical risk/error:** $\hat{R}(f)=\frac{1}{n}\mathop{\Sigma}\limits_{i=1}^nL\{f(x_i),y_i\} $ (the estimator of R(f))<br>

# 2.PAC
+ **concept of complexity:**
    1. Computational complexity: storage and time
    2. Sample complexity: amount of training data needed to provide statistical guarantees
        for successful learning, that is sample size
+ **PAC:** Probably Approximately Correct learning
    + **Definition:** concept class $\mathcal{F}$ is PAC-learnable if there exists a learning algorithm or procedure L such that
       +  for any f ∈ F, and all ε > 0,δ > 0, and all distributions D, \mathbb{P}_{S\sim\mathcal{D}^n}\{R(f_S)\le\epsilon\}\ge1-\delta 
       + sample S has size p(1\backslash\epsilon,1\backslash\delta)  for some polynomial p 
       + Sometimes,L is requested to run in a time polynomial in the size of S.
+ **Probably** is determined by the confidence 1 − δ
+ **Approximately Correct** is determined by the accuracy 1 − ε

# 3.Model selection
+ **CV:** easy to overfit when choosing variable, better to use in prediction process
+ **AIC:** Model selection inconsistent
+ **BIC:** consistant,  better to use when choosing variable
![2](2.png)
+ **Regularzation:**
    1. Ridge regression:$ \hat{\beta}_{ridge}=argmin_{\beta}\|Y-\mathbb{X}\beta\|_2^2+\lambda\|\beta\|_2^2$
       <br>for linear regression:the solution $\hat{\beta}_{ridge}=(\mathbb{X}^T\mathbb{X}+\lambda I)^{-1}\mathbb{X}^TY$ <br>
       <br>look deeper: let $\mathbb{X}=UDV^T$ , <br>
<br>$\hat{\beta}_{OLS}=VD^{-1}U^TY= \mathop{\Sigma}_{j=1}^pv_j(\frac{1}{d_j})u_j^TY$ <br>
<br>$\hat{\beta}_{ridge}=V(D^2+\lambda I)^{-1}DU^TY=\mathop{\Sigma}_{j=1}^pv_j(\frac{d_j}{d_j^2+\lambda})u_j^TY <br>$
<br>$\mathbb{X}\hat{\beta}_{OLS}=UU^TY= \qquad\mathop{\Sigma}_{j=1}^pu_ju_j^TY <br>$
<br>$\mathbb{X}\hat{\beta}_{ridge}=UD(D^2+\lambda I)^{-1}DU^TY=\mathop{\Sigma}_{j=1}^pu_j(\frac{d_j^2}{d_j^2+\lambda})u_j^TY <br>$
    2. Lasso :$ \hat{\beta}_{lasso}=argmin_{\beta}\|Y-\mathbb{X}\beta\|_2^2+\lambda\|\beta\|_1 $
    3. $\|\cdot\|_p$ :  convexity and model selection(sparsity)
    ![3](3.png)
    

# 4.Maximum Entropy Models and Logistic Regression

## 4.1 information theory
+ **entropy:** $H(X)=-\mathbb{E}\{logp(X)\}=-\mathop{\Sigma}\limits_{x\in \mathcal{X}}p(x)logp(x)\le logN $
                <br> Entropy measures the uncertainty of X <br>
                <br> Maximal for uniform distribution by Jensen’s inequality<br>

## 4.2  Logistic Regression(逻辑回归)
For a binary dependent variable, the  LR is a **conditional probability distribution**：
<br>$P(Y=1|x)=\frac{exp(wx+b)}{1+exp(wx+b)}=\frac{1}{1+exp(-(wx+b))} $<br><br>$ P(Y=0|x)=\frac{1}{1+exp(wx+b)}$ <br>
<br>the probability of the event happening： p ，<br>
<br>the probability of the event not happening: 1-p ,<br>
<br>the odds(几率) of the event: $\frac{p}{1-p} $,<br>
<br>the log odds/logit function of the event: $log\frac{p}{1-p} $,<br>
<br>from above, we can get:  $log\frac{P(Y=1|x)}{1-P(Y=1|x)}=wx+b$,it's a linear regression model.<br>
the parameter estimation：MLE
<br>$P(Y=1|x)=\pi(x)        ,     P(Y=0|x)=1-\pi(x) $<br>
<br>the likelihood function: $\mathop{\prod}\limits_{i=1}^N[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}$ <br>
<br>negative  log likelihood function: \begin{equation} \begin{array}{l} &L(w) &=-\mathop{\Sigma}\limits_{i=1}^N[y_ilog\pi(x_i)+(1-y_i)log(1-\pi(x_i))]\\ & &=-\mathop{\Sigma}\limits_{i=1}^N[y_ilog\frac{\pi(x_i)}{1-\pi(x_i)}+log(1-\pi(x_i))]\\ & &=-\mathop{\Sigma}\limits_{i=1}^N[y_i(wx_i+b)-log(1+exp(wx_i+b))] \end{array} \end{equation} <br>
let $z_i=wx_i+b , L(w)=-\mathop{\Sigma}\limits_{i=1}^N[y_iz_i-log(1+exp(z_i))] $ ,that is to say  the loss function of LR is  logarithmic loss function.
**Here is another form:** $L(w)=\mathop{\Sigma}\limits_{i=1}^Nlog(1+exp(-y_iz_i)) $
**this two form is equivalent** i.e., $-y_iz_i+log(1+exp(z_i))=log(1+exp(-y_iz_i)) $
<br>the left:$ y=\{1,0\} $,the right: $y=\{1,-1\} $<br>
<br>when $y_i=1$, it's  obvious :$-log(exp(z_i))+log(1+exp(z_i))=log\frac{1+exp(z_i)}{exp(z_i)}=log(1+exp(-z_i))$<br>
<br>when $y_i=0 (left), y_i=-1 (right)$,  it's also equivalent.<br>
<br>The problem turns to optimization problem, using gradient decent or Quasi-Newton method(拟牛顿法) .<br>

## 4.3 Maximum Entropy Model
+ **maximum entropy theory:**
<br>Maximum entropy theory is a principle in probability model learning. It believes that the model whose entropy is the largest is the best model of all possible probability models. Intuitively, it's thought that the model to select should meet the constraints. In the case of not having more information, the uncertain parts are all with equal probability.<br>
+ **Maximum Entropy Model:** 
    + **defination:** Suppose the set of all models that meet the constrains: $\mathcal{C}=\{P\in\mathcal{P}|E_p(f_i)=E_{\tilde{p}}(f_i),i=1,2,\cdots,n\} $,the conditional entropy defined on conditional probability distribution : $H(P)=-\mathop{\Sigma}\limits_{x,y}\tilde{P}(x)P(y|x)logP(y|x) $,the model in $\mathcal{C}$ whose conditional entropy H(P) is biggest is called maximum entropy model.
    + the problem equals to the optimization problem:
\begin{equation} \begin{array}{ll} \max\limits_{\mathcal{P}\in \mathcal{C}}&H(P)=-\mathop{\Sigma}\limits_{x,y}\tilde{P}(x)P(y|x)logP(y|x)\\ s.t.& E_P(f_i)=E_{\tilde{P}(f_i)},i=1,2,\cdots,n \\ &\mathop{\Sigma}\limits_{y}P(y|x)=1 \end{array} \end{equation} 

# 5. Ensemble Methods
  根据个体学习器的生成方式，目前的集成学习算法大致分为两类：个体学习器间存在强依赖关系、必须串行生成的方法，代表Boosting；个体学习器间不存在强依赖关系、可同时生成的并行化方法，代表Bagging以及random Forest。
## 5.1Boosting

 + **3 ingredients**: base learner;  learning rate;  number of base learner
<br>The most popular algorithm is Adaboost:<br>

+ **Algorithm Adaboost:**

 1. Initialize the weights distribution of training data:
<br>$D_1=(w_{11},w_{12},\cdots,w_{1n}) ,  w_{1i}=1/n,i=1,2,\cdots,n $<br>
 2.  for b =1,2,....,B
     <br>(a) learning with the weighted data, get the  base classifier: $G_b(x)$<br>
    <br> (b)calculate the error rate   of $G_b(x)$ :  $e_b=P(G_b(x_i)\neq y_i)=\mathop{\Sigma}\limits_{i=1}^nw_{bi}\mathbb{I}(G_b(x_i)\neq y_i)$ <br>
     <br>(c)compute the coefficient of $G_b(x) $:  $\alpha_{b}=\frac{1}{2}log(\frac{1-e_b}{e_b})  ( e_b\leq1/2\Rightarrow \alpha_b\geq0 )$<br>
     <br>(d)update the weights of training data: <br>
 <br>$w_{b+1,i}=\frac{w_{bi}}{Z_b}exp(-\alpha_by_iG_b(x_i))=\left\{ \begin{array} &\frac{w_{bi}}{Z_b}exp(-\alpha_b),&y_i=G_b(x_i)\\ \frac{w_{bi}}{Z_b}exp(\alpha_b),&y_i\neq G_b(x_i) \end{array} \right . $
<br>$Z_b=\mathop{\Sigma}\limits_{i=1}^nw_{bi}exp(-\alpha_by_iG_b(x_i)) $<br>
 3. get the final classifier : $G(x)=sign(f(x))=sign(\mathop{\Sigma}\limits_{b=1}^B\alpha_bG_b(x)) $

## 5.2 Bagging
+ based on bootstrap
+ Bagging Algorithm：Instead of having B separate training sets, we train on B bootstrap draws:$ \hat{f_1}(X),\cdots,\hat{f_B}(X) $，and then average them: $\hat{f_{bag}}(X)=\frac{1}{B}\mathop{\Sigma}\limits_{b=1}^B\hat{f_b}(X) $
+ bagging trees:
     - Choose a large number B;
     - For each b = 1,...,B, grow an unpruned tree on the b^{th} bootstrap drawn
         from the data;
     - Average all these trees together. 
+ **the difference of bagging and boosting:**
<br>**Bagging** is a procedure for taking a low bias, high variance and reduce the risk via averaging.<br>
**Boosting** is a procedure that has high bias but low variance.