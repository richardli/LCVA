% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{lcm_fit}
\alias{lcm_fit}
\title{Internal function to fit the nested latent class model on training data}
\usage{
lcm_fit(
  X,
  Y,
  Group,
  X_test,
  Y_test,
  Group_test,
  N_train,
  N_test,
  S,
  C,
  K,
  G,
  alpha_pi,
  alpha_eta,
  a_omega,
  b_omega,
  nu_phi,
  a_gamma,
  b_gamma,
  nu_tau,
  Nitr,
  thin,
  similarity,
  sparse,
  verbose
)
}
\arguments{
\item{X}{a n by p matrix of the symptoms with 0 being absent, 1 being present, and NA being missing.}

\item{Y}{a vector of length n of the causes-of-death, coded into 1 to C, where C is the total number of all cause.}

\item{Group}{a vector of length n of the domain indicators, coded into 1 to G, where G is the total number of all domains.}

\item{X_test}{currently not used.}

\item{Y_test}{currently not used.}

\item{Group_test}{currently not used.}

\item{N_train}{size of training data, n.}

\item{N_test}{currently not used.}

\item{S}{number of symptoms.}

\item{C}{number of causes.}

\item{K}{number of latent classes within each cause-of-death.}

\item{G}{number of training domains.}

\item{alpha_pi}{Concentration parameter for the training domain CSMF prior.}

\item{alpha_eta}{currently not used.}

\item{a_omega}{Shape parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior.}

\item{b_omega}{Rate parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior.}

\item{nu_phi}{Shape2 parameter of the beta distribution for the class-dependent response probabilities.}

\item{a_gamma}{Shape1 parameter of the gamma prior for the baseline response probabilities.}

\item{b_gamma}{Shape2 parameter of the gamma prior for the baseline response probabilities.}

\item{nu_tau}{Shape2 parameter of the sparsity level in the response probabilities.}

\item{Nitr}{number of iterations to run in each MCMC chain.}

\item{thin}{number of draws to sample per one saved.}

\item{similarity}{shrinkage model for the testing domain mixing weights. Currently not used.}

\item{sparse}{binary indicator of whether to encourage latent profiles to be sparse}
}
\description{
Internal function to fit the nested latent class model on training data
}
\examples{
message("See ?LCVA.train")
}
