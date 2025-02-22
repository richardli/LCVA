#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <stdio.h>
#include <math.h> 

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;
using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;


//' Calculating the log of sum of exponentiation vector
//' @param x a numeric vector
//' @return log(sum(exp(x[1]) + exp(x[2]) + ...))
//' @examples
//' logsumexp(c(1, 2, 3))
//' log(sum(exp(c(1, 2, 3))))
//' @export
// [[Rcpp::export]]
double logsumexp(arma::vec &x){
  double xs = max(x);    
  double p = xs + log(sum(exp(x - xs)));
  return(p);
}
//' Sample from log of unnormalized probability vector
//' @param x a vector of log of unnormalized probability vector
//' @return a sample from 0 to length(x)-1, with probabilities exp(x)/sum(exp(x))
//' @examples
//' x <- rep(0, 10000)
//' prob <- c(1, 2, 3, 4)
//' logprob <- log(prob) 
//' for(i in 1:10000){
//'   x[i] <- sample_log_prob(logprob) 
//' }
//' round(table(x) / 10000, 2)
//' @export
//' 
// [[Rcpp::export]]
int sample_log_prob(arma::vec &x){
  double xs = max(x);    
  arma::vec p = cumsum(exp(x - (xs + log(sum(exp(x - xs))))));
  int len = p.size();
  double u = runif(1, 0.0, 1.0)(0);
  int i;
  for(i = 0; i < len; i++){
    if(u <= p(i)){
        return(i);
    }   
  }
  return(-1);
}
//' Sample from a column of the log of unnormalized probability matrix
//' @param x a matrix where each column contains the log of a unnormalized probability vector
//' @param id column index, starting from 0.
//' @return a sample from 0 to length(x)-1, with probabilities exp(x[, id + 1])/sum(exp(x, id + 1))
//' @examples
//' x <- rep(0, 10000)
//' prob <- cbind(c(1, 2, 3, 4), c(4, 3, 2, 1))
//' logprob <- log(prob) 
//' for(i in 1:10000){
//'   x[i] <- sample_log_prob_matrix_col(logprob, 0) 
//' }
//' table(x) / 10000
//' for(i in 1:10000){
//'   x[i] <- sample_log_prob_matrix_col(logprob, 1) 
//' }
//' round(table(x) / 10000, 2)
//' @export
//' 
// [[Rcpp::export]]
int sample_log_prob_matrix_col(arma::mat &x, int id){
    arma::vec y = x.col(id);
    int k = sample_log_prob(y);
    return(k);
}

//' Sample from log of unnormalized probability matrix
//' @param x a matrix where each cell contains the log of a unnormalized probability matrix.
//' @return the row and column index of a sample of the cells in the matrix. Index starting from 0.
//' @examples
//' x <- matrix(0, 10000, 2)
//' prob <- cbind(c(1, 2, 3, 4), c(4, 3, 2, 1))
//' logprob <- log(prob) 
//' for(i in 1:10000){
//'   x[i, ] <- sample_log_prob_matrix(logprob) 
//' }
//' round(table(paste(x[, 1], x[, 2], sep = "-")) / 10000, 2)
//' prob / sum(prob)
//' @export
//' 
// [[Rcpp::export]]
arma::vec sample_log_prob_matrix(arma::mat &x){
  arma::vec y(x.n_cols * x.n_rows);
  int i, j;
  int tmp = 0;
  for(i = 0; i < x.n_rows; i ++){
    for(j = 0; j < x.n_cols; j++){
        y(tmp) = x(i, j);
        tmp += 1;       
    }
  }
  int k  = sample_log_prob(y);  
  arma::vec out(2);
  // k = 9, x ~ 3*4, should return (2, 1)
  // column index 
  out(1) = k % x.n_cols;
  // row index
  out(0) = (k - out(1)) / x.n_cols;
  return(out);
}

//' Sample from a probability vector
//' @param x a vector of a probability vector
//' @return a sample from 0 to length(x)-1, with probabilities x / sum(x).
//' @examples
//' x <- rep(0, 10000)
//' prob <- c(.1, .2, .3, .4)
//' for(i in 1:10000){
//'   x[i] <- sample_prob(prob) 
//' }
//' round(table(x) / 10000, 2)
//' @export
//' 
// [[Rcpp::export]]
int sample_prob(arma::vec &x){
  int len = x.size();
  arma::vec p = cumsum(x);
  double u = runif(1, 0.0, 1.0)(0);
  int i;
  for(i = 0; i < len; i++){
    if(u <= p(i)){
        return(i);
    }   
  }
  return(-1);
}
 
//' Sample from the Dirichlet distribution
//'
//' @param a vector of Dirichlet concentration parameter
//' @return a sample from Dirichlet(a).  
//' @examples
//' alpha <- c(1, 2, 3, 4)
//' x <- matrix(0, 10000, 4)
//' for(i in 1:10000){
//'   x[i, ] <- sample_Dirichlet(alpha) 
//' }
//' round(apply(x, 2, mean), 2)
//' @export
//' 
// [[Rcpp::export]]
arma::vec sample_Dirichlet(arma::vec &a){
  int len = a.size();
  arma::vec rg(len);
  double sum = 0;
  int i = 0;
  for(i = 0; i < len; i++){
    if(a(i) == 0){
        rg(i) = 0;
    }else{
        rg(i) = Rcpp::rgamma(1, a(i), 1)(0);
        sum += rg(i);        
    }
  }
  arma::vec out = rg / sum;
  return(out);
}


//' Internal function to fit the nested latent class model on training data
//' 
//' @param X a n by p matrix of the symptoms with 0 being absent, 1 being present, and NA being missing.
//' @param Y a vector of length n of the causes-of-death, coded into 1 to C, where C is the total number of all cause.
//' @param Group a vector of length n of the domain indicators, coded into 1 to G, where G is the total number of all domains.
//' @param X_test currently not used.
//' @param Y_test currently not used.
//' @param Group_test currently not used.
//' @param N_train size of training data, n. 
//' @param N_test currently not used.
//' @param S number of symptoms.
//' @param C number of causes.
//' @param K number of latent classes within each cause-of-death.
//' @param G number of training domains.
//' @param alpha_pi Concentration parameter for the training domain CSMF prior.  
//' @param alpha_eta currently not used.
//' @param a_omega Shape parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior.
//' @param b_omega Rate parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior.
//' @param nu_phi Shape2 parameter of the beta distribution for the class-dependent response probabilities.
//' @param nu_tau Shape2 parameter of the sparsity level in the response probabilities.
//' @param a_gamma Shape1 parameter of the gamma prior for the baseline response probabilities.
//' @param b_gamma Shape2 parameter of the gamma prior for the baseline response probabilities.
//' @param Nitr number of iterations to run in each MCMC chain.
//' @param thin number of draws to sample per one saved.
//' @param similarity shrinkage model for the testing domain mixing weights. Currently not used.
//' @param sparse binary indicator of whether to encourage latent profiles to be sparse
//' 
//' @examples
//' message("See ?LCVA.train")
// [[Rcpp::export]]
SEXP lcm_fit(SEXP X, SEXP Y, SEXP Group, 
         SEXP X_test, SEXP Y_test, SEXP Group_test,
         int N_train, int N_test, int S, int C, int K, int G, 
         double alpha_pi, double alpha_eta, double a_omega, double b_omega, 
         double nu_phi, SEXP a_gamma, SEXP b_gamma, double nu_tau, 
         int Nitr, int thin, int similarity, int sparse) {

    // organize input
    arma::mat X1 = as<mat>(X);
    arma::vec Y1 = as<vec>(Y);
    arma::vec G1 = as<vec>(Group);
    arma::mat X0 = as<mat>(X_test);
    arma::vec Y0 = as<vec>(Y_test);

    arma::vec G0 = as<vec>(Group_test);
    arma::mat a_cj = as<mat>(a_gamma);
    arma::mat b_cj = as<mat>(b_gamma);

    // Start from 0
    G1 = G1 - 1;
    G0 = G0 - 1;
    Y1 = Y1 - 1;
    Y0 = Y0 - 1; // code Y_test = -1 if unknown


    // latent indicators
    arma::vec Z1(N_train);
    arma::vec Z0(N_test);
    arma::vec G_latent(N_test);
    // latent probability
    arma::cube phi(C, K, S);
    arma::cube logphi(C, K, S);
    arma::cube log_1_minus_phi(C, K, S);
    arma::mat gamma(C, S);    
    arma::cube delta(C, K, S);
    arma::vec tau(C);
    arma::mat eta(G, C);
    arma::mat config(G, C);

    // mixture parameters
    arma::cube lambda(C, K, G);
    arma::cube V(C, K, G);
    arma::mat V_test(C, K);
    arma::mat omega(C, G);
    arma::mat lambda_test(C, K);
    arma::mat pi(C, G);
    arma::vec pi_test(C);
    String itrname("itr0");

    // mixture parameters for the matrix sticking process
    arma::mat U(C, K);
    arma::mat W(G, K);
    // double omega_U;
    // double omega_W;
    // arma::mat nR_ck(C, K);
    // arma::mat nS_gk(G, K);
    // arma::mat nR0_ck(C, K);
    // arma::mat nS0_gk(G, K);
    // arma::vec nS_gk_test(K);     
    // arma::vec nS0_gk_test(K); 

    // parameters for the separate process for the test data
    arma::vec omega_test(C);
    // arma::vec W_test(K); 

    // output
    int Nout = Nitr;
    arma::mat Z1_out(Nout, N_train);
    arma::mat Z0_out(Nout, N_test);
    arma::mat Y0_out(Nout, N_test);
    arma::mat G_latent_out(Nout, N_test);
    arma::mat tau_out(Nout, C);
    arma::cube eta_out(Nout, G, C);
    arma::cube pi_out(Nout, C, G);
    arma::mat pi_test_out(Nout, C);
    arma::mat loglik(Nout, N_train);
    // arma::vec omega_U_out(Nout);
    // arma::vec omega_W_out(Nout);
    List phi_out, gamma_out, lambda_out, lambda_test_out;  

    // count arrays
    arma::cube n_ckg(C, K, G);
    arma::cube n_ckj_0(C, K, S);
    arma::cube n_ckj_1(C, K, S);
    arma::mat n_ck(C, K);
    arma::mat n_cg(C, G); 
    arma::mat n0_cj(C, S);
    arma::mat n1_cj(C, S);

    arma::mat n_ck_test(C, K);
    arma::vec n_c_test(C); 
    arma::mat n_g_latent(G, C); 

    int itr, itr_save, i, j, c, k, l, g, zy, itr_tmp;
    int itr_show = 500;
    if(Nitr < 2001) itr_show = 200;
    if(Nitr < 1001) itr_show = 100;
    arma::vec pz(K), pg(G), py(C), probrs(3);
    arma::mat pzg(K, G);
    arma::mat pyg(C, K);
    arma::mat pzyg(C*K, G);
    arma::cube px(N_test, C, K);
    arma::vec index2(2);
    double tmp, sumV, sumN, tmp0, tmp1, sumlambda;
    double tol = 0.000001;
    int stick_break = 1;

    arma::vec Y0_known(N_test);
    for(i = 0; i < N_test; i++){
        if(Y0(i) < 0){
            Y0_known(i) = 0;
        }else{
            Y0_known(i) = 1;
        }
    }

    // 
    // Initialization
    // 
    if(stick_break > 0){
        for(g = 0; g < G; g++){
            for(c = 0; c < C; c++){
                omega(c, g) = Rcpp::rgamma(1, a_omega, 1/b_omega)(0);
                //  all but the last stick
                sumlambda = 0.0;
                for(k = 0; k < K - 1; k++){
                    V(c, k, g) = Rcpp::rbeta(1, 1, omega(c, g))(0);
                    if(V(c, k, g) < tol){
                         V(c, k, g) = tol; 
                    }
                    if(V(c, k, g) > 1 - tol){
                         V(c, k, g) = 1 - tol; 
                    }
                    lambda(c, k, g) = log(V(c, k, g));
                    for(l = 0; l < k; l++){
                        lambda(c, k, g) += log(1 - V(c, l, g));
                    }
                    sumlambda = sumlambda + exp(lambda(c, k, g));
                }
                // last stick
                V(c, K-1, g) = 1;
                if(sumlambda > 1 - tol){
                    lambda(c, K-1, g) = log(tol);
                }else{
                    lambda(c, K-1, g) = log(1 - sumlambda);
                }
            }
        }
        for(c = 0; c < C; c++) omega_test(c) = Rcpp::rgamma(1, a_omega, 1/b_omega)(0);
    }


    if(lambda.has_inf()){
        Rcout << "lambda has inf\n";
    }

    config.zeros();
    for(i = 0; i < N_train; i++){
        config(G1(i), Y1(i)) += 1;
    }
        // Rcout <<config << "\n";

    for(c = 0; c < C; c++){
        tmp = 0;
        for(g = 0; g < G; g++){
            tmp = tmp + config(g, 0);
        }
        for(g = 0; g < G; g++){
            eta(g, c) = config(g, c) / tmp;
        }
    }
    lambda_test.zeros();
    for(c = 0; c < C; c++){
        for(k = 0; k < K; k++){
            for(g = 0; g < G; g++){
                lambda_test(c, k) += exp(lambda(c, k, g)) * eta(g, c);
            }
            lambda_test(c, k) = log(lambda_test(c, k));
        }
    }

    if(lambda_test.has_inf()){
        Rcout << "lambda test has inf\n";
    }
   
    for(c = 0; c < C; c++){
        for(j = 0; j < S; j++){
            gamma(c, j) = Rcpp::rbeta(1, a_cj(c, j), b_cj(c, j))(0);
            for(k = 0; k < K; k++){
                phi(c, k, j) = Rcpp::rbeta(1, 1, nu_phi)(0);
                logphi(c, k, j) = log(phi(c, k, j));
                log_1_minus_phi(c, k, j) = log(1 - phi(c, k, j));
            }
        }
    }
    for(c = 0; c < C; c++){
        tau(c) = rbeta(1, 1, nu_tau)(0);
    }

    // initialize latent states
    pi = pi.ones();
    pi_test = pi_test.zeros();
    // if(common_pi == 0){
        for(i = 0; i < N_train; i++){
            pi(Y1(i), G1(i)) ++;
        }

        for(g = 0; g < G; g++){
            tmp = 0;
            for(c = 0; c < C; c++){
                tmp += pi(c, g);
            }
            for(c = 0; c < C; c++){
                pi(c, g) = pi(c, g) / tmp; 
                pi_test(c) += pi(c, g) / (G + 0.0);
            }
        }
    // }else{
    //     pi = pi.ones();
    //     for(i = 0; i < N_train; i++){
    //         pi(Y1(i), 0) += 1 / (N_train + C + 0.0);
    //     }
    //     for(c = 0; c < C; c++){
    //         for(g = 1; g < G; g++){
    //             pi(c, g) = pi(c, 0);
    //         }
    //         pi_test(c) = pi(c, 0);
    //     }

    // }

    for(i = 0; i < N_test; i++){
        if(Y0_known(i) == 0){
            Y0(i) = sample_prob(pi_test);
        }
        pg = eta.col(Y0(i));
        G_latent(i) = sample_prob(pg);
    }
    Rcout << "Start posterior sampling\n";
    for(itr_save = 0; itr_save < Nitr; itr_save ++){
        // Rcout << ".";
        itr_tmp = (itr_save + 1) % itr_show;
        if(itr_tmp == 0){
          itr_tmp = (itr_save + 1) * thin;
          Rcout << "Iteration " << itr_tmp << " completed.\n";
        }

        // 
        // Sampler starts here
        // 
        for(itr = 0; itr < thin; itr ++){
           
            // ------------------------//
            // Sample Z in training
            // ------------------------//
            n_ckj_1 = n_ckj_1.zeros();
            n_ckj_0 = n_ckj_0.zeros();
            n_ckg = n_ckg.zeros();
            n_ck = n_ck.zeros();
            n_cg = n_cg.zeros();
            n_c_test = n_c_test.zeros();
            n_ck_test = n_ck_test.zeros();

            for(i = 0; i < N_train; i++){
                for(k = 0; k < K; k++){
                    pz(k) = lambda(Y1(i), k, G1(i));
                }
                pyg.zeros();
                for(j = 0; j < S; j++){
                    if(!ISNA(X1(i, j))){
                        pyg += X1(i, j) * logphi.slice(j) + (1 - X1(i, j)) * log_1_minus_phi.slice(j);
                    }
                }
                // for(j = 0; j < S; j++){
                //     if(!ISNA(X1(i, j))){
                //         for(k = 0; k < K; k++){
                //             pz(k) += logphi(Y1(i), k, j)*X1(i, j);
                //             pz(k) += log_1_minus_phi(Y1(i), k, j) * (1-X1(i, j));
                //         }
                //     }
                // }

                // compute log likelihood
                py.zeros();
                tmp = 0;
                for(c = 0; c < C; c++){
                    for(k = 0; k < K; k++){
                        py(c) += exp(lambda(c, k, G1(i)) + pyg(c, k));
                    }
                    py(c) *= pi(c, G1(i));
                    tmp += py(c);
                }
                loglik(itr_save, i) = log(py(Y1(i))) - log(tmp);

                // sample Z
                for(k = 0; k < K; k++){
                    pz(k) += pyg(Y1(i), k);
                }
                Z1(i) = sample_log_prob(pz);
                n_ck(Y1(i), Z1(i)) += 1;
                n_ckg(Y1(i), Z1(i), G1(i)) += 1;
                n_cg(Y1(i), G1(i)) += 1;
                for(j = 0; j < S; j++){
                    if(!ISNA(X1(i, j))){
                        n_ckj_1(Y1(i), Z1(i), j) += X1(i, j);
                        n_ckj_0(Y1(i), Z1(i), j) += 1-X1(i, j);
                    }
                }
            }


            // ----------------------------//
            // sample Z and Y0 in testing
            // ---------------------------//
            px.zeros();
            n_g_latent.zeros();
            G_latent.zeros();
            for(i = 0; i < N_test; i++){
                if(Y0_known(i) == 0){
                    pzyg.zeros();

                    if(G0(i) < 0 && similarity >= 1){
                        for(c = 0; c < C; c++){
                            for(k = 0; k < K; k++){
                                for(g = 0; g < G; g++){
                                   pzyg(k * C + c, g) += lambda(c, k, g) + log(pi_test(c)) + log(eta(g, c));
                                }
                            }
                        }
                    }else if(G0(i) < 0 && similarity == 0){
                        for(c = 0; c < C; c++){
                            for(k = 0; k < K; k++){
                                 pzyg(k * C + c, 0) += lambda_test(c, k) + log(pi_test(c));
                             }
                         }
                    }else{
                        for(c = 0; c < C; c++){
                            for(k = 0; k < K; k++){
                                pzyg(k * C + c, G0(i)) += lambda(c, k, G0(i)) + log(pi(c, G0(i)));
                            }
                        }
                    }

                    for(j = 0; j < S; j++){
                        if(!ISNA(X0(i, j))){
                            for(c = 0; c < C; c++){
                                for(k = 0; k < K; k++){
                                    pzyg.row(k * C + c) += logphi(c, k, j)*X0(i, j);
                                    pzyg.row(k * C + c) += log_1_minus_phi(c, k, j) * (1-X0(i, j));
                                    px(i, c, k) += logphi(c, k, j)*X0(i, j);
                                    px(i, c, k) += log_1_minus_phi(c, k, j) * (1-X0(i, j));
                                }
                            }
                        }
                    }
                    index2.zeros();
                    if(G0(i) < 0 && similarity >= 1){
                        index2 = sample_log_prob_matrix(pzyg);
                    }else if(G0(i) < 0 && similarity == 0){
                        index2(0) = sample_log_prob_matrix_col(pzyg, 0);
                    }else{
                        index2(0) = sample_log_prob_matrix_col(pzyg, G0(i));
                    }

                     
                    zy = index2(0);
                    Y0(i) = zy % C;
                    Z0(i) = (zy - Y0(i)) / C;
                    // sample latent group membership 
                    if(G0(i) < 0 && similarity >= 1){
                        G_latent(i) = index2(1);
                        n_g_latent(G_latent(i), Y0(i)) ++;
                    }
                }else{
                    pzg.zeros();
                    for(k = 0; k < K; k++){
                        if(G0(i) < 0 & similarity >= 1){
                            for(g = 0; g < G; g++){    
                                pzg(k, g) = lambda(Y0(i), k, g) + log(eta(g, Y0(i)));
                            }
                        }else if(G0(i) < 0 & similarity == 0){
                            pzg(k, 0) = lambda_test(Y0(i), k);
                        }else{
                            pzg(k, G0(i)) = lambda(Y0(i), k, G0(i));
                        }
                        for(j = 0; j < S; j++){
                            if(!ISNA(X0(i, j))){
                                pzg.row(k) += logphi(Y0(i), k, j)*X0(i, j);
                                pzg.row(k) += log_1_minus_phi(Y0(i), k, j) * (1-X0(i, j));
                            }
                        }
                    }

                    if(G0(i) < 0 & similarity >= 1){
                        index2 = sample_log_prob_matrix(pzg);
                    }else if(G0(i) < 0 & similarity == 0){
                        index2(0) = sample_log_prob_matrix_col(pzg, 0);
                    }else{
                        index2(0) = sample_log_prob_matrix_col(pzg, G0(i));
                    }
                    Z0(i) = index2(0);
                    if(G0(i) < 0 && similarity >= 1){
                        G_latent(i) = index2(1);
                        n_g_latent(G_latent(i), Y0(i)) ++;
                    }
                }

                if(G0(i) < 0){
                    n_ck_test(Y0(i), Z0(i)) += 1;
                    n_c_test(Y0(i)) += 1;
                }
                // add in test counts
                for(j = 0; j < S; j++){
                    if(!ISNA(X0(i, j))){
                        n_ckj_1(Y0(i), Z0(i), j) += X0(i, j);
                        n_ckj_0(Y0(i), Z0(i), j) += 1-X0(i, j);
                    }
                }

            }

            // ------------------------------------------------//
            // Sample V and omega in training, independent stick
            // ------------------------------------------------//
           if(stick_break > 0){
                for(g = 0; g < G; g++){
                    for(c = 0; c < C; c++){

                        sumV = 0.0;
                        sumlambda = 0.0;
                        sumN = n_cg(c, g);
                        for(k = 0; k < K - 1; k++){
                            sumN = sumN - n_ckg(c, k, g);
                            V(c, k, g) = Rcpp::rbeta(1,  n_ckg(c, k, g)+1.0, omega(c, g) + sumN + 0.0)(0);
                            sumV = sumV + log(1 - V(c, k, g));
                            if(V(c, k, g) < tol){
                                 V(c, k, g) = tol; 
                            }
                            if(V(c, k, g) > 1 - tol){
                                 V(c, k, g) = 1 - tol; 
                            }
                            lambda(c, k, g) = log(V(c, k, g));
                            for(l = 0; l < k; l++){
                                lambda(c, k, g) += log(1 - V(c, l, g));
                            }
                            sumlambda = sumlambda + exp(lambda(c, k, g));
                        }
                        if(sumlambda > 1 - tol){
                            lambda(c, K-1, g) = log(tol);
                        }else{
                            lambda(c, K-1, g) = log(1 - sumlambda);
                        }

                        omega(c, g) = Rcpp::rgamma(1, a_omega + K - 1, 1/(b_omega - sumV))(0);
                    }
                }

            // ------------------------------------------------//
            // Sample V and omega in training, matrix stick
            // ------------------------------------------------//

           }

           // ------------------------------------------------//
          // Sample eta and lambda in testing, weighted sum
           // ------------------------------------------------//
           if(similarity >= 1 && N_test > 0){
                if(similarity == 1){
                    // make eta_g across all c
                    for(g = 0; g < G; g++){
                        pg(g)  = alpha_eta;
                        for(c = 0; c < C; c++){
                            pg(g) += n_g_latent(g, c);
                        }
                    }
                    eta.col(0) =  sample_Dirichlet(pg);
                    for(c = 1; c < C; c++){
                        for(g = 0; g < G; g++){
                            eta(g, c) = eta(g, 0);
                        }
                    }
                }else if(similarity == 2){
                    // make eta_g unique for each c
                    for(c = 0; c < C; c++){
                        for(g = 0; g < G; g++){
                            pg(g) = alpha_eta * config(g, c) +  n_g_latent(g, c);
                        }
                        eta.col(c) = sample_Dirichlet(pg);
                    }
                }
                lambda_test.zeros();
                for(c = 0; c < C; c++){
                    for(k = 0; k < K; k++){
                        for(g = 0; g < G; g++){
                            lambda_test(c, k) += exp(lambda(c, k, g)) * eta(g, c);
                        }
                        lambda_test(c, k) = log(lambda_test(c, k));
                    }
                }
           }else if(N_test > 0){
            eta = eta.zeros();
            G_latent = G_latent.zeros();

            // ------------------------------------------------//
            // Sample lambda in testing, from prior, indep stick
            // ------------------------------------------------//
            if(stick_break > 0){
                for(c = 0; c < C; c++){
                    sumV = 0.0;
                    sumlambda = 0.0;
                    sumN = n_c_test(c); 
                    for(k = 0; k < K - 1; k++){
                        sumN = sumN - n_ck_test(c, k);  
                        V_test(c, k) = Rcpp::rbeta(1,  n_ck_test(c, k)+1.0, omega_test(c) + sumN + 0.0)(0);
                        sumV = sumV + log(1 - V_test(c, k));
                        if(V_test(c, k) < tol){
                             V_test(c, k) = tol; 
                        }
                        if(V_test(c, k) > 1 - tol){
                             V_test(c, k) = 1 - tol; 
                        }
                        lambda_test(c, k) = log(V_test(c, k));
                        for(l = 0; l < k; l++){
                            lambda_test(c, k) += log(1 - V_test(c, l));
                        }
                        sumlambda = sumlambda + exp(lambda_test(c, k));
                    }
                    if(sumlambda > 1 - tol){
                        lambda_test(c, K-1) = log(tol);
                    }else{
                        lambda_test(c, K-1) = log(1 - sumlambda);
                    }
                    omega_test(c) = Rcpp::rgamma(1, a_omega + K - 1, 1/(b_omega - sumV))(0);
                }
           }
        }


            // ------------------------------------------------//
            // Sample delta, tau, and phi
            // ------------------------------------------------//       
            for(c = 0; c < C; c++){  
                tmp = 0;  
                for(k = 0; k < K; k++){
                    for(j = 0; j < S; j++){
                        if(sparse == 1){
                            tmp0 = log(1 - tau(c)) + 
                                   log(gamma(c, j)) * n_ckj_1(c,k,j) +  
                                   log(1-gamma(c, j)) * n_ckj_0(c,k,j) + 
                                   lgamma(1) + lgamma(nu_phi) - lgamma(1 + nu_phi);
                            tmp1 = log(tau(c)) + 
                                   lgamma(1 + n_ckj_1(c,k,j)) +
                                   lgamma(nu_phi + n_ckj_0(c,k,j)) - 
                                   lgamma(1 + nu_phi + n_ckj_1(c,k,j) + n_ckj_0(c,k,j));
                            // e^x / (e^x + e^y) = 1 / (1 + e^{y - x})       
                            delta(c, k, j) = Rcpp::rbinom(1, 1, 1 / (1 + exp(tmp0 - tmp1)))(0);
                        }else{
                            delta(c, k, j) = 1;
                        }
                                // Rcout << tmp1 << tmp0 << "\n";
                                // if(delta.has_nan()){
                                //      Rcout << tmp0 <<" "<< tmp1 << "\n";
                                //      Rcout << (1 - tau(c)) <<" "<< gamma(c, j) <<" "<<  n_ckj_1(c,k,j) << "\n";
                                // }
                        tmp += delta(c, k, j);
                        if(delta(c, k, j) > 0.5){
                            phi(c, k, j) = Rcpp::rbeta(1, 1 + n_ckj_1(c,k,j), nu_phi + n_ckj_0(c,k,j))(0);  
                        }else{
                            phi(c, k, j) = gamma(c, j);
                        }
                        logphi(c, k, j) = log(phi(c, k, j));
                        log_1_minus_phi(c, k, j) = log(1 - phi(c, k, j));
                    }
                }
                tau(c) = Rcpp::rbeta(1, 1 + tmp, nu_tau + K * S - tmp)(0);
                // Rcout << "-----" << tmp <<" "<< K * S - tmp << "\n";
            } 

            // ------------------------------------------------//
            // Sample gamma
            // ------------------------------------------------//             
            n0_cj = n0_cj.zeros();
            n1_cj = n1_cj.zeros();
            for(i = 0; i < N_train; i++){
                for(j = 0; j < S; j++){
                    if(!ISNA(X1(i, j))){
                        n1_cj(Y1(i), j) += X1(i, j) * (1 - delta(Y1(i), Z1(i), j));
                        n0_cj(Y1(i), j) += (1 - X1(i, j)) * (1 - delta(Y1(i), Z1(i), j));
                    }
                }
            }
            // add in test counts
             for(i = 0; i < N_test; i++){
                for(j = 0; j < S; j++){
                    if(!ISNA(X0(i, j))){
                        n1_cj(Y0(i), j) += X0(i, j) * (1 - delta(Y0(i), Z0(i), j));
                        n0_cj(Y0(i), j) += (1 - X0(i, j)) * (1 - delta(Y0(i), Z0(i), j));
                    }
                }
            }


            for(c = 0; c < C; c++){    
                for(j = 0; j < S; j++){
                    // tmp0 = a_cj(c, j);
                    // tmp1 = b_cj(c, j);
                    // for(k = 0; k < K; k++){
                    //     tmp0 += n_ckj_1(c,k,j) * (1-delta(c,k,j));
                    //     tmp1 += (n_ck(c,k) - n_ckj_1(c,k,j)) * (1-delta(c,k,j));
                    // }
                    gamma(c, j) = Rcpp::rbeta(1, a_cj(c, j) + n1_cj(c, j), b_cj(c, j) + n0_cj(c, j))(0); 
                                // if(gamma.has_nan()){
                                //      Rcout << tmp0 <<" "<< tmp1 << "\n";
                                // }
                }
            }
             for(c = 0; c < C; c++){    
                for(k = 0; k < K; k++){
                    for(j = 0; j < S; j++){
                        if(delta(c, k, j) < 0.5){
                            phi(c, k, j) = gamma(c, j);
                            logphi(c, k, j) = log(phi(c, k, j));
                            log_1_minus_phi(c, k, j) = log(1 - phi(c, k, j));
                        }
                        // avoid numerical issue with extreme distribution?
                        // if(phi(c, k, j) < tol){
                        //      phi(c, k, j) = tol; 
                        // }
                        // if(phi(c, k, j) > 1 - tol){
                        //      phi(c, k, j) = 1 - tol; 
                        // }
                    }
                }
            }

            // // Sample Y0
            // for(i = 0; i < N_test; i++){
            //     py.zeros();
            //    for(c = 0; c < C; c++){
            //         pz.zeros();
            //         for(k = 0; k < K; k++){
            //             for(j = 0; j < S; j++){
            //                 if(!ISNA(X0(i, j))){
            //                     pz(k) += log(phi(c, k, j))*X0(i, j);
            //                     pz(k) += log(1-phi(c, k, j)) * (1-X0(i, j));
            //                 }
            //             }
            //             if(G0(i) < 0){
            //                pz(k) += lambda_test(c, k);
            //             }else{
            //                pz(k) += lambda(c, k, G0(i));
            //             }
            //         }
                    
            //         py(c) = logsumexp(pz);
            //         if(G0(i) < 0){
            //             py(c) += log(pi_test(c));
            //         }else{
            //             py(c) += log(pi(c, G0(i)));
            //         }
            //     }
            //     Y0(i) = sample_log_prob(py);
            // }
            
            // ------------------------------------------------//
            // Sample pi in test
            // ------------------------------------------------// 
            n_cg.zeros();
            n_c_test.zeros(); 
            for(i = 0; i < N_train; i++){
                n_cg(Y1(i), G1(i))++;
            }
            for(i = 0; i < N_test; i++){
                if(G0(i) < 0){
                    n_c_test(Y0(i))++;
                }else{
                    n_cg(Y0(i), G0(i))++;
                }
            }
            // if(common_pi == 0){
                for(g = 0; g < G; g++){
                    py.zeros();
                    for(c = 0; c < C; c++){
                        py(c) = alpha_pi + n_cg(c, g);
                    }
                    pi.col(g) = sample_Dirichlet(py);
                } 
                py.zeros();
                for(c = 0; c < C; c++){
                    py(c) = alpha_pi + n_c_test(c);
                }
                pi_test = sample_Dirichlet(py);
            // }else{
            //     py.zeros();
            //     for(c = 0; c < C; c++){
            //         py(c) = alpha_pi;
            //         for(g = 0; g < G; g++){
            //             py(c) += n_cg(c, g);
            //         }
            //     } 
            //     pi.col(0) = sample_Dirichlet(py);
            //     for(c = 0; c < C; c++){
            //         for(g = 1; g < G; g++){
            //           pi(c, g) = pi(c, 0);
            //         }
            //         pi_test(c) = pi(c, 0);
            //     }
            // }

           

        }


        // // compute log likelihood
        // for(i = 0; i < N_train; i++){
        //     py.zeros();
        //     pyg.zeros();
        //     tmp = 0;
        //     for(j = 0; j < S; j++){
        //         if(!ISNA(X1(i, j))){
        //             pyg += X1(i, j) * logphi.slice(j) + (1 - X1(i, j)) * log_1_minus_phi.slice(j);
        //         }
        //     }
        //     for(c = 0; c < C; c++){
        //         for(k = 0; k < K; k++){
        //             py(c) += exp(lambda(c, k, G1(i)) + pyg(c, k));
        //         }
        //         // for(j = 0; j < S; j++){
        //         //     if(!ISNA(X1(i, j))){
        //         //         for(k = 0; k < K; k++){
        //         //             pz(k) += logphi(c, k, j)*X1(i, j);
        //         //             pz(k) += log_1_minus_phi(c, k, j) * (1-X1(i, j));
        //         //         }
        //         //     }
        //         // }
        //         // for(k = 0; k < K; k++){
        //         //     py(c) += exp(pz(k));
        //         // }
        //         py(c) *= pi(c, G1(i));
        //         tmp += py(c);
        //     }
        //     loglik(itr_save, i) = log(py(Y1(i))) - log(tmp);
        // }


        //  save results
        itrname = "itr" + std::to_string(itr_save);
        phi_out(itrname) = phi;
        gamma_out(itrname) = gamma;
        lambda_out(itrname) = lambda;
        Z1_out.row(itr_save) = Z1.t();
        tau_out.row(itr_save) = tau.t();
        pi_out.row(itr_save) = pi;   
        if(N_test > 0){
            Z0_out.row(itr_save) = Z0.t();
            Y0_out.row(itr_save) = Y0.t();
            G_latent_out.row(itr_save) = G_latent.t();
            pi_test_out.row(itr_save) = pi_test.t();   
            eta_out.row(itr_save) = eta;
        }
        if(similarity == 0){
            lambda_test_out(itrname) = lambda_test;
        }

    }

    List out;
    out("X") = X1;
    out("Y") = Y1 + 1;
    out("Group") = G1 + 1;
    out("S") = S;
    out("C") = C;
    out("K") = K;
    out("G") = G;
    out("alpha_pi") = alpha_pi;
    out("alpha_eta") = alpha_eta;
    out("a_omega") = a_omega;
    out("b_omega") = b_omega;
    out("nu_phi") = nu_phi;
    out("nu_tau") = nu_tau;
    out("phi") = phi_out;
    out("gamma") = gamma_out;
    out("loglambda") = lambda_out;
    out("Z1") = Z1_out + 1;
    out("tau") = tau_out;
    out("eta") = eta_out;
    out("pi") = pi_out;
    out("loglik") = loglik;
    out("config") = config;
   
    if(N_test > 0){
        out("X_test") = X0;
        out("Group_test") = G0 + 1;
        out("Z0") = Z0_out + 1;
        out("G_latent") = G_latent_out + 1;
        out("pi_test") = pi_test_out;
        out("Y_test") = Y0_out + 1;
        if(similarity == 0){
            out("loglambda_test") = lambda_test_out;
        }
     }
  return out;
}






//' Internal function to predict with the nested latent class model
//' 
//' @param X_test a n by p matrix of the symptoms with 0 being absent, 1 being present, and NA being missing.
//' @param Y_test a vector of length n of the causes-of-death, coded into 0 to C, where C is the total number of all cause and 0 indicates unknown cause of death.  
//' @param Group_test a vector of length n of the domain indicators, coded into 0 to G, where G is the total number of all domains and 0 indicates a new target domain. 
//' @param config_train a matrix of counts for all domain-cause combinations.
//' @param N_test number of deaths to assign a cause to.
//' @param S number of symptoms.
//' @param C number of causes.
//' @param K number of latent classes within each cause-of-death.
//' @param G number of training domains.
//' @param itr_draws vector of iteration indices to use from the training posterior draws.
//' @param alpha_pi_vec vector of the concentration parameters for the target domain CSMF.
//' @param alpha_eta  concentration parameter for the domain or domain-cause mixture prior. Only used for multi-domain models. 
//' @param a_omega Shape parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior.  
//' @param b_omega Rate parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior. 
//' @param lambda_fit posterior draws of the training mix weights.
//' @param phi_fit posterior draws of the response probabilities.
//' @param pi_fit posterior draws of the training CSMF.
//' @param pi_init initial values of the target CSMF.
//' @param Nitr number of iterations to run in each MCMC chain.
//' @param similarity shrinkage model for the testing domain mixing weights. Currently not used.
//' 
//' @examples
//' message("See ?LCVA.pred")
// [[Rcpp::export]]
SEXP lcm_pred(SEXP X_test, SEXP Y_test, SEXP Group_test, SEXP config_train,
         int N_test, int S, int C, int K, int G, SEXP itr_draws,
         SEXP alpha_pi_vec, double alpha_eta, double a_omega, double b_omega, 
         arma::field<arma::cube> lambda_fit, arma::field<arma::cube> phi_fit, arma::cube pi_fit, SEXP pi_init,
         int Nitr, int similarity, int return_x_given_y) {
  
    arma::mat X0 = as<mat>(X_test);
    arma::vec Y0 = as<vec>(Y_test);
    arma::vec G0 = as<vec>(Group_test);
    arma::vec pi_test = as<vec>(pi_init);
    arma::mat config = as<mat>(config_train);
    arma::vec alpha_pi = as<vec>(alpha_pi_vec);
    arma::vec itr_use = as<vec>(itr_draws);
    // Start from 0
    G0 = G0 - 1;
    Y0 = Y0 - 1; // code Y_test = -1 if unknown

    // latent indicators
    arma::vec Z0(N_test);
    arma::vec G_latent(N_test);
    // latent probability
    arma::mat pi(C, G);
    arma::cube logphi(C, K, S);
    arma::cube log_1_minus_phi(C, K, S);
    arma::vec logpi_test(C);
    arma::mat logeta(G, C);
    arma::cube lambda(C, K, G);
    arma::mat eta(G, C);
    // arma::mat config(G, C);
    arma::mat lambda_test(C, K);
    arma::vec omega_test(C);
    arma::mat V_test(C, K);
    String itrname("itr0");


    // output
    int Nout = Nitr;
    arma::mat Z0_out(Nout, N_test);
    arma::mat Y0_out(Nout, N_test);
    arma::mat G_latent_out(Nout, N_test);
    arma::cube eta_out(Nout, G, C);
    arma::mat pi_test_out(Nout, C);
    arma::cube x_given_y_out(Nout, N_test, C);
    List lambda_test_out;  

    // count arrays
    arma::cube n_ckg(C, K, G);
    arma::cube n_ckj_0(C, K, S);
    arma::cube n_ckj_1(C, K, S);
    arma::mat n_ck(C, K);
    arma::mat n_cg(C, G); 
    arma::mat n0_cj(C, S);
    arma::mat n1_cj(C, S);

    arma::mat n_ck_test(C, K);
    arma::vec n_c_test(C); 
    arma::mat n_g_latent(G, C); 

    int itr_save, i, j, c, k, l, g, zy, s, itr_tmp;
    int itr_show = 500;
    if(Nitr < 2001) itr_show = 200;
    if(Nitr < 1001) itr_show = 100;
    arma::vec pz(K), pg(G), py(C), probrs(3);
    arma::mat pyg(C, K);
    arma::mat pzg(K, G);
    arma::mat pzyg(C*K, G);
    // arma::cube px(N_test, C, K);
    arma::vec index2(2);
    double sumV, sumN, sumlambda, lambda_ck;
    double tol = 0.000001;

    arma::vec Y0_known(N_test);
    for(i = 0; i < N_test; i++){
        if(Y0(i) < 0){
            Y0_known(i) = 0;
        }else{
            Y0_known(i) = 1;
        }
    }
    // omega_test
    for(c = 0; c < C; c++) omega_test(c) = Rcpp::rgamma(1, a_omega, 1/b_omega)(0);

    for(c = 0; c < C; c++){
        for(g = 0; g < G; g++){
            eta(g, c) = 1 / (G + 0.0);
            logeta(g, c) = log(eta(g, c));
        }
        logpi_test(c) = log(pi_test(c));
    }
    lambda_test.zeros();
    for(c = 0; c < C; c++){
        for(k = 0; k < K; k++){
            for(g = 0; g < G; g++){
                lambda_test(c, k) += exp(lambda_fit(0)(c, k, g)) * eta(g, c);
            }
            lambda_test(c, k) = log(lambda_test(c, k));
        }
    }

    if(lambda_test.has_inf()){
        Rcout << "lambda test has inf\n";
    }

    Rcout << "Start posterior sampling\n";
    for(itr_save = 0; itr_save < Nitr; itr_save ++){

        itr_tmp = (itr_save + 1) % itr_show;
        if(itr_tmp == 0){
          // itr_tmp = itr_save * thin;
          Rcout << "Iteration " << itr_save + 1 << " completed.\n";
        }

        i = itr_use(itr_save);
        for(c = 0; c < C; c++){
            for(k = 0; k < K; k++){
                for(s = 0; s < S; s++){
                    logphi(c, k, s) = log(phi_fit(i)(c, k, s));
                    log_1_minus_phi(c, k, s) = log(1 - phi_fit(i)(c, k, s));
                }
                for(g = 0; g < G; g++){
                    lambda(c, k, g) = lambda_fit(i)(c, k, g);
                }
            }
            for(g = 0; g < G; g++){
                pi(c, g) = pi_fit(i, c, g);
            }
        }
        // if(common_pi == 1){
        //     for(c = 0; c < C; c++){
        //         pi_test(c) = pi_fit(i, c, 0);
        //     }
        // }
        // ----------------------------//
        // sample Z and Y0 in testing
        // ---------------------------//
        // px.zeros();
        n_g_latent.zeros();
        G_latent.zeros();
        n_c_test = n_c_test.zeros();
        n_ck_test = n_ck_test.zeros();
       
        for(i = 0; i < N_test; i++){
            if(Y0_known(i) == 0){
                pzyg.zeros();

                if(G0(i) < 0 && similarity >= 1){
                    for(c = 0; c < C; c++){
                        for(k = 0; k < K; k++){
                            for(g = 0; g < G; g++){
                               pzyg(k * C + c, g) += lambda(c, k, g) + logpi_test(c) + logeta(g, c);
                            }
                        }
                    }
                }else if(G0(i) < 0 && similarity == 0){
                    for(c = 0; c < C; c++){
                        for(k = 0; k < K; k++){
                             pzyg(k * C + c, 0) += lambda_test(c, k) + logpi_test(c);
                         }
                     }
                // if data from existing domains, not used for now
                }else{
                    for(c = 0; c < C; c++){
                        for(k = 0; k < K; k++){
                            pzyg(k * C + c, G0(i)) += lambda(c, k, G0(i)) + log(pi(c, G0(i)));
                        }
                    }
                }
                pyg.zeros();
                for(j = 0; j < S; j++){
                    if(!ISNA(X0(i, j))){
                        pyg += X0(i, j) * logphi.slice(j) + (1 - X0(i, j)) * log_1_minus_phi.slice(j);
                    }
                }
                for(c = 0; c < C; c++){
                    for(k = 0; k < K; k++){
                        pzyg.row(k * C + c) += pyg(c, k);
                        // pzyg.row(k * C + c) += logphi(c, k, j)*X0(i, j);
                        // pzyg.row(k * C + c) += log_1_minus_phi(c, k, j) * (1-X0(i, j));
                    }
                }
                    
                
                
                index2.zeros();
                if(G0(i) < 0 && similarity >= 1){
                    index2 = sample_log_prob_matrix(pzyg);
                }else if(G0(i) < 0 && similarity == 0){
                    index2(0) = sample_log_prob_matrix_col(pzyg, 0);
                }else{
                    index2(0) = sample_log_prob_matrix_col(pzyg, G0(i));
                }

                 
                zy = index2(0);
                Y0(i) = zy % C;
                Z0(i) = (zy - Y0(i)) / C;
                // sample latent group membership 
                if(G0(i) < 0 && similarity >= 1){
                    G_latent(i) = index2(1);
                    n_g_latent(G_latent(i), Y0(i)) ++;
                }
                if(return_x_given_y){
                    // First extract the right lambda_{c, k}
                    // The X|Y probability is 
                    //    \sum_k lambda_{c, k} * P(X | Y = c, Z = k) 
                    //  = \sum_k lambda_{c, k} * pyg(c, k)
                    // in the first case, lambda_{ck} = \sum \lambda_{ckg} * eta(gc)
                    if(G0(i) < 0 && similarity >= 1){
                        for(c = 0; c < C; c++){
                            x_given_y_out(itr_save, i, c) = 0;
                            lambda_ck = 0;
                            for(k = 0; k < K; k++){
                                for(g = 0; g < G; g++){
                                    lambda_ck += exp(lambda(c, k, g) + logeta(g, c));
                                }
                                x_given_y_out(itr_save, i, c) += lambda_ck * exp(pyg(c, k));
                            }
                        }
                    }else if(G0(i) < 0 && similarity == 0){
                        for(c = 0; c < C; c++){
                            x_given_y_out(itr_save, i, c) = 0;
                            for(k = 0; k < K; k++){
                                x_given_y_out(itr_save, i, c) += exp(pyg(c, k) + lambda_test(c, k));
                            }
                        }
                    }else{
                        for(c = 0; c < C; c++){
                            x_given_y_out(itr_save, i, c) = 0;
                            for(k = 0; k < K; k++){
                                x_given_y_out(itr_save, i, c) += exp(pyg(c, k) + lambda(c, k, G0(i)));
                            }
                        }
                    }
                }
            }else{
                pzg.zeros();
                for(k = 0; k < K; k++){
                    if(G0(i) < 0 & similarity >= 1){
                        for(g = 0; g < G; g++){    
                            pzg(k, g) = lambda(Y0(i), k, g) + logeta(g, Y0(i));
                        }
                    }else if(G0(i) < 0 & similarity == 0){
                        pzg(k, 0) = lambda_test(Y0(i), k);
                    }else{
                        pzg(k, G0(i)) = lambda(Y0(i), k, G0(i));
                    }
                    for(j = 0; j < S; j++){
                        if(!ISNA(X0(i, j))){
                            pzg.row(k) += logphi(Y0(i), k, j)*X0(i, j);
                            pzg.row(k) += log_1_minus_phi(Y0(i), k, j) * (1-X0(i, j));
                        }
                    }
                }

                if(G0(i) < 0 & similarity >= 1){
                    index2 = sample_log_prob_matrix(pzg);
                }else if(G0(i) < 0 & similarity == 0){
                    index2(0) = sample_log_prob_matrix_col(pzg, 0);
                }else{
                    index2(0) = sample_log_prob_matrix_col(pzg, G0(i));
                }
                Z0(i) = index2(0);
                if(G0(i) < 0 && similarity >= 1){
                    G_latent(i) = index2(1);
                    n_g_latent(G_latent(i), Y0(i)) ++;
                }
            }

 
            if(G0(i) < 0){
                n_ck_test(Y0(i), Z0(i)) += 1;
                n_c_test(Y0(i)) += 1;
            }

        }

       // ------------------------------------------------//
       // Sample eta and lambda in testing, weighted sum
       // ------------------------------------------------//
       if(similarity >= 1 && N_test > 0){
            if(similarity == 1){
                // make eta_g across all c
                for(g = 0; g < G; g++){
                    pg(g)  = alpha_eta;
                    for(c = 0; c < C; c++){
                        pg(g) += n_g_latent(g, c);
                    }
                }
                eta.col(0) =  sample_Dirichlet(pg);
                for(g = 0; g < G; g++){
                    logeta(g, 0) = log(eta(g, 0));
                }
                for(c = 1; c < C; c++){
                    for(g = 0; g < G; g++){
                        eta(g, c) = eta(g, 0);
                        logeta(g, c) = log(eta(g, 0));
                    }
                }
            }else if(similarity == 2){
                // make eta_g unique for each c
                for(c = 0; c < C; c++){
                    for(g = 0; g < G; g++){
                        pg(g) = alpha_eta * config(g, c) +  n_g_latent(g, c);
                    }
                    eta.col(c) = sample_Dirichlet(pg);
                    for(g = 0; g < G; g++){
                        logeta(g, c) = log(eta(g, c));
                    }
                }
            }
            lambda_test.zeros();
            for(c = 0; c < C; c++){
                for(k = 0; k < K; k++){
                    for(g = 0; g < G; g++){
                        lambda_test(c, k) += exp(lambda(c, k, g)) * eta(g, c);
                    }
                    lambda_test(c, k) = log(lambda_test(c, k));
                }
            }
       }else if(N_test > 0){
            eta = eta.zeros();
            G_latent = G_latent.zeros();

            // ------------------------------------------------//
            // Sample lambda in testing, from prior, indep stick
            // ------------------------------------------------//

            for(c = 0; c < C; c++){
                sumV = 0.0;
                sumlambda = 0.0;
                sumN = n_c_test(c); 
                for(k = 0; k < K - 1; k++){
                    sumN = sumN - n_ck_test(c, k);  
                    V_test(c, k) = Rcpp::rbeta(1,  n_ck_test(c, k)+1.0, omega_test(c) + sumN + 0.0)(0);
                    sumV = sumV + log(1 - V_test(c, k));
                    if(V_test(c, k) < tol){
                         V_test(c, k) = tol; 
                    }
                    if(V_test(c, k) > 1 - tol){
                         V_test(c, k) = 1 - tol; 
                    }
                    lambda_test(c, k) = log(V_test(c, k));
                    for(l = 0; l < k; l++){
                        lambda_test(c, k) += log(1 - V_test(c, l));
                    }
                    sumlambda = sumlambda + exp(lambda_test(c, k));
                }
                if(sumlambda > 1 - tol){
                    lambda_test(c, K-1) = log(tol);
                }else{
                    lambda_test(c, K-1) = log(1 - sumlambda);
                }
                omega_test(c) = Rcpp::rgamma(1, a_omega + K - 1, 1/(b_omega - sumV))(0);
            }
        }
        // ------------------------------------------------//
        // Sample pi in test
        // ------------------------------------------------// 
        n_c_test.zeros(); 
        for(i = 0; i < N_test; i++){
            if(G0(i) < 0){
                n_c_test(Y0(i))++;
            }
        }
        // if(common_pi == 0){
            py.zeros();
            for(c = 0; c < C; c++){
                py(c) = alpha_pi(c) + n_c_test(c);
            }
            pi_test = sample_Dirichlet(py);
            for(c = 0; c < C; c ++){
                logpi_test(c) = log(pi_test(c));
            }
        // }
        //  save results
        itrname = "itr" + std::to_string(itr_save);
        if(N_test > 0){
            Z0_out.row(itr_save) = Z0.t();
            Y0_out.row(itr_save) = Y0.t();
            G_latent_out.row(itr_save) = G_latent.t();
            pi_test_out.row(itr_save) = pi_test.t();   
            eta_out.row(itr_save) = eta;
        }
        if(similarity == 0){
            lambda_test_out(itrname) = lambda_test;
        }

    }

    List out;
    if(N_test > 0){
        out("X_test") = X0;
        out("Group_test") = G0 + 1;
        out("Z0") = Z0_out + 1;
        out("G_latent") = G_latent_out + 1;
        out("pi_test") = pi_test_out;
        out("Y_test") = Y0_out + 1;
        out("eta") = eta_out;
        out("alpha_pi") = alpha_pi;
        if(similarity == 0){
            out("loglambda_test") = lambda_test_out;
        }
        if(return_x_given_y == 1){
            out("x_given_y_prob") = x_given_y_out;
        }
     }
  return out;
}