# Count Data
HSLC 5210
David Rudoler 

---
## What is count data?
- A non-negative integer: $y \in \mathbb{N_0} = \{0,1,2,...\}$
- **Example:** hospitalizations, doctor visits, co-morbidities. 
- The distribution of count variables mass at non-negative values; thus, we make different parametric assumptions about these data than we would in a linear context (i.e., where there is a normally distributed outcome).  ^ab2f3a

---
## Poisson distribution & model
The probability mass function for the Poisson distribution is: 

$$Pr(Y = y) = \frac{e^{-\lambda}\lambda^y}{y!}$$
where $\lambda$ is the rate parameter. The mean and variance of the Poisson distribution are equal to $\lambda$. The **Poisson model** estimates $\lambda = exp(x'\beta)$. This ensures that all values of $\lambda >0$

---
```R
# Randomly draw from a poisson distribution with lambda = 1. 
set.seed(1234)
x <- rpois(n = 10000, lambda = 1)

summary(x)

# cumulative frequency
cs <- data.frame(table(x))
cs$cumulative <- (cs$Freq/sum(cs$Freq))*100

#histogram 
hist(x)
```

The mean of the distribution is equal to 1 and the variance is 0.982. 

| Value | Frequency | Percent |
| ----- | --------- | ------- |
| 0     | 3,638     | 36.38   |
| 1     | 3,739     | 37.39   |
| 2     | 1,809     | 18.09   |
| 3     | 633       | 6.33    |
| 4     | 152       | 1.52    |
| 5     | 28        | 0.28    |
| 6     | 1         | 0.01    |

![[rpois(lambda = 1).png]]

When the mean is small (as in the case above), then the values of the distribution will cluster in a few distinct values; in the distribution above, > 70% of the values are zero or 1. 

If we increase the value of $\lambda$, the proportion of zeros decreases and the distribution spreads over a greater number of values. If we set $\lambda = 5$, then the mean of the distribution is 5.029 and the variance is 4.941. The distribution is as follows: 

| Value | Frequency | Percent |
| ----- | --------- | ------- |
| 0     | 66        | 0.66    |
| 1     | 325       | 3.25    |
| 2     | 813       | 8.13    |
| 3     | 1347      | 13.47   |
| 4     | 1770      | 17.70   |
| 5     | 1788      | 17.88   |
| 6     | 1508      | 15.08   |
| 7     | 1030      | 10.30   |
| 8     | 701       | 7.01    |
| 9     | 342       | 3.42    |
| 10    | 170       | 1.70    |
| 11    | 84        | 0.84    |
| 12    | 33        | 0.33    |
| 13    | 10        | 0.10    |
| 14    | 9         | 0.09    |
| 15    | 3         | 0.03    |
| 16    | 1         | 0.01    | 
![[rpois(lambda = 5).png]]
### Poisson Model Estimation 
```R
library(dplyr)
library(sandwich) #robust SEs
library(msm) #delta method

# load data 
nhanes <- readRDS("~/Sync/Work/Teaching/HLSC5210/data/nhanes_1718.rds")

# Model number of times in past 30 days with 4-5 drinks
## filter out non-resposes
### Note: important to deal with missing values in normal cicumstances
dat <- nhanes_1718 %>% 
  filter(alq170 != 777 & alq170 != 999)

summary(dat$alq170)

pois <- glm(alq170 ~ riagendr + ridageyr, family = "poisson", data = dat)
cov_pois <- vcovHC(pois, type = "HC0")
std_err <- sqrt(diag(cov_pois))
r_est <- cbind(Estimate = coef(pois), 
               "Robust SE" = std_err, 
               "Pr(>|z|)" = 2 * pnorm(abs(coef(pois)/std_err), lower.tail = F),
                       LB = coef(pois) - 1.96 * std_err,
                       UB = coef(pois) + 1.96 * std_err)
r_est
```


| var       | estimate | robust SE | p-value | 95% CI       |
| --------- | -------- | --------- | ------- | ------------ |
| intercept | 1.84     | 0.20      | <0.001  | 1.44, 2.24   |
| Female    | -0.75    | 0.11      | <0.001  | -0.97, -0.53 |
| Age       | -0.013   | 0.003     | <0.001  | -0.02, -.007 | 

The coefficients are in log counts. We can generate incidence rate ratios by exponentiating and using the "[delta method](https://cran.r-project.org/web/packages/modmarg/vignettes/delta-method.html)" to generate standard errors. 

```R
# delta method applied to each x variable (k = 3) in our model. 
se <- deltamethod(list(~ exp(x1), ~ exp(x2), ~ exp(x3)), 
                  coef(pois), cov_pois)

## Exponentiate coefficients (remove p-values)
coef_exp <- exp(r_est[,-3])
#replace SEs with those generated using deltamethod
coef_exp[,"Robust SE"] <- se

coef_exp
```

| var       | estimate | robust SE | 95% CI       |
| --------- | -------- | --------- | ------------ |
| intercept | 2.968    | 0.404     | 2.273, 3.875 |
| Female    | 0.472    | 0.054     | 0.378, 0.590 |
| Age       | 0.987    | 0.003     | 0.982, 0,993 | 

We can interpret these coefficients as rate ratios. For example, the rate of drinking 4-5 drinks per occasion in the last 30 days 100(1-0.472) = 52% lower for women, compared to men. And, for every one year of age, the rate decreases by 100(1-0.987) = 1.3%. 


## Negative binomial data 
- Since the mean and variance of the Poisson distribution are equal, when we use the Poisson model we make the assumption that the conditional mean and variance of our outcome are equal. This is the so called **equidispersion assumption**. This assumption is often violated in practice. 
- When our data is **over-dispersed**, the conditional variance is larger than our conditional mean.  The variance is scaled by a value $\nu$ that is drawn from a gamma distribution, typically with a mean = 1 and a variance equal to $\alpha$. 
- When over-dispersion is present, we can apply a negative binomial model, where $E(y|\lambda, \alpha) = \lambda$ and $Var(y|\lambda, \alpha) = \lambda(1 + \alpha \lambda)$. This is the typical quadratic variance (known as NB2). There is an alternative linear variance (NB1) where $Var(y|\lambda, \alpha) = (1 + \alpha)\lambda$. 
- As $\alpha$ approaches zero, the negative binomial model reduces to the Poisson model. 

---
```R
# Randomly draw from a gamma distribution with shape = 1, and scale = 1). 
set.seed(1234)
nu <- rgamma(n = 10000, shape = 1, scale = 1)

# draw from poisson-gamma mixture (negative binomial)
nb <- rpois(n = 10000, lambda = nu)

# cumulative frequency
cs <- data.frame(table(nb))
cs$cumulative <- (cs$Freq/sum(cs$Freq))*100

#histogram 
hist(nb)

```

The mean of the distribution is equal to 0.99 and the variance is 1.87. The distribution is as follows: 

| Value | Frequency | Percent |
| ----- | --------- | ------- |
| 0     | 4962      | 49.62   |
| 1     | 2547      | 25.47   |
| 2     | 1271      | 12.71   |
| 3     | 639       | 6.39    |
| 4     | 304       | 3.04    |
| 5     | 143       | 1.43    |
| 6     | 72        | 0.72    |
| 7     | 32        | 0.32    |
| 8     | 14        | 0.14    |
| 9     | 6         | 0.06    |
| 10    | 8         | 0.08    |
| 11    | 1         | 0.01    |
| 12    | 1         | 0.01    |

![[nb.png]]
### Negative binomial model estimation 

```R
library(MASS) # for NB regression 

nbr <- glm.nb(alq170 ~ riagendr + ridageyr, data = dat)

# exponentiate the estimates to obtain IRRs
exp((est <- cbind(Estimate = coef(nbr), confint(nbr))))
```

| var       | estimate | robust SE | 95% CI        |
| --------- | -------- | --------- | ------------- |
| intercept | 7.425    | 1.670     | 4.778, 11.537 |
| Female    | 0.465    | 0.056     | 0.366, 0.590  |
| Age       | 0.984    | 0.003     | 0.978, 0,990  |

### Test for over-dispersion 

```R
pchisq(2 * (logLik(nbr) - logLik(pois)), df = 1, lower.tail = F)
```

We reject the null that the two conditional variances are equal in the Poisson and the negative binomial model. This suggests that the negative binomial is the more appropriate estimator. 

## Other Count Data Models 
Other count data models deal with cases where there are a high number of zeros (e.g., [[Hurdle Model]], [[Zero-inflated model]]) which are under predicted by standard Poisson and Negative Binomial estimators. Finite mixture models stratify the sample into latent classes and estimate separate count data estimators for each distinct class. Panel data models allow us to generate more efficient estimates, and potentially take into account time-invariant unobserved confounders (like [[HLSC 5210 - Lecture Longitudinal Data]])
### Hurdle Model 
A hurdle model can be used when we believe the process generating the zeros is different than the process generating the positive values. In particular, we believe there is some "hurdle" that must be crossed before positive values for the outcome can be accumulated. 

The classic example is doctor visits. In order to have a positive count of visits, one must first cross the hurdle of getting access to a doctor. Thus, it would make sense in this context to estimate the process in two parts: one for the probability of accessing a doctor at all, and another for the number of doctor visits a person has. Not using this approach may lead to the use of count data models that do not account well for the zeros in our sample. A hurdle model can be estimated in R as follows: 

```R
# using the same NHANES data as we used before. 
# Package that lets you run a hurdle model
library(pscl)

# hurdle model with negative binomial applied to positive values and logit applied to 0s and 1s. 
hurd <- hurdle(alq170 ~ riagendr + ridageyr, 
               dist = "negbin", 
               zero.dist = "binomial",
               data = nhanes)
summary(hurd)
```

This will produce results form two different models: one for the binary outcome and one for the positive values. We can interpret them the same as we did for [[logistic regression]] and the count data models discussed above. 

### Zero-inflated  
Another option for data where there are a high proportion of zeros is the Zero-inflated model. Similar to the Hurdle Model, the Zero-inflated model also assumes that the zeros are generated via a different data generating process than the positive values. 

The key difference relates to the theory around the decision-making process of the study subjects. 
- Hurdle models are motivated by sequential decision-making (i.e., first I choose to seek out a doctor, then I decide how many visits I want to have with that doctor).
- Zero-inflated models are not (i.e., first I choose to seek out a doctor, then I decide whether to see that doctor or not and how many times I want to see them). 

Zero-inflated models can be estimated in R as follows: 

```R
# Using the same NHANES data as above 
zinf <- zeroinfl(alq170 ~ riagendr + ridageyr, data = nhanes)
summary(zinf)
```

### Details on other count data models 
- [Finite mixture model](https://cran.r-project.org/web/packages/flexmix/vignettes/flexmix-intro.pdf)
- [Panel Poisson model](https://rpubs.com/cuborican/xtpoisson)