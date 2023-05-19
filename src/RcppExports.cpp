// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// logsumexp
double logsumexp(arma::vec& x);
RcppExport SEXP _LCVA_logsumexp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(logsumexp(x));
    return rcpp_result_gen;
END_RCPP
}
// sample_log_prob
int sample_log_prob(arma::vec& x);
RcppExport SEXP _LCVA_sample_log_prob(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_log_prob(x));
    return rcpp_result_gen;
END_RCPP
}
// sample_log_prob_matrix_col
int sample_log_prob_matrix_col(arma::mat& x, int id);
RcppExport SEXP _LCVA_sample_log_prob_matrix_col(SEXP xSEXP, SEXP idSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type id(idSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_log_prob_matrix_col(x, id));
    return rcpp_result_gen;
END_RCPP
}
// sample_log_prob_matrix
arma::vec sample_log_prob_matrix(arma::mat& x);
RcppExport SEXP _LCVA_sample_log_prob_matrix(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_log_prob_matrix(x));
    return rcpp_result_gen;
END_RCPP
}
// sample_prob
int sample_prob(arma::vec& x);
RcppExport SEXP _LCVA_sample_prob(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_prob(x));
    return rcpp_result_gen;
END_RCPP
}
// sample_Dirichlet
arma::vec sample_Dirichlet(arma::vec& a);
RcppExport SEXP _LCVA_sample_Dirichlet(SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_Dirichlet(a));
    return rcpp_result_gen;
END_RCPP
}
// lcm_fit
SEXP lcm_fit(SEXP X, SEXP Y, SEXP Group, SEXP X_test, SEXP Y_test, SEXP Group_test, int N_train, int N_test, int S, int C, int K, int G, double alpha_pi, double alpha_eta, double a_omega, double b_omega, double nu_phi, SEXP a_gamma, SEXP b_gamma, double nu_tau, int Nitr, int thin, int similarity, int sparse);
RcppExport SEXP _LCVA_lcm_fit(SEXP XSEXP, SEXP YSEXP, SEXP GroupSEXP, SEXP X_testSEXP, SEXP Y_testSEXP, SEXP Group_testSEXP, SEXP N_trainSEXP, SEXP N_testSEXP, SEXP SSEXP, SEXP CSEXP, SEXP KSEXP, SEXP GSEXP, SEXP alpha_piSEXP, SEXP alpha_etaSEXP, SEXP a_omegaSEXP, SEXP b_omegaSEXP, SEXP nu_phiSEXP, SEXP a_gammaSEXP, SEXP b_gammaSEXP, SEXP nu_tauSEXP, SEXP NitrSEXP, SEXP thinSEXP, SEXP similaritySEXP, SEXP sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type X(XSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Y(YSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Group(GroupSEXP);
    Rcpp::traits::input_parameter< SEXP >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Y_test(Y_testSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Group_test(Group_testSEXP);
    Rcpp::traits::input_parameter< int >::type N_train(N_trainSEXP);
    Rcpp::traits::input_parameter< int >::type N_test(N_testSEXP);
    Rcpp::traits::input_parameter< int >::type S(SSEXP);
    Rcpp::traits::input_parameter< int >::type C(CSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_pi(alpha_piSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_eta(alpha_etaSEXP);
    Rcpp::traits::input_parameter< double >::type a_omega(a_omegaSEXP);
    Rcpp::traits::input_parameter< double >::type b_omega(b_omegaSEXP);
    Rcpp::traits::input_parameter< double >::type nu_phi(nu_phiSEXP);
    Rcpp::traits::input_parameter< SEXP >::type a_gamma(a_gammaSEXP);
    Rcpp::traits::input_parameter< SEXP >::type b_gamma(b_gammaSEXP);
    Rcpp::traits::input_parameter< double >::type nu_tau(nu_tauSEXP);
    Rcpp::traits::input_parameter< int >::type Nitr(NitrSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< int >::type similarity(similaritySEXP);
    Rcpp::traits::input_parameter< int >::type sparse(sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(lcm_fit(X, Y, Group, X_test, Y_test, Group_test, N_train, N_test, S, C, K, G, alpha_pi, alpha_eta, a_omega, b_omega, nu_phi, a_gamma, b_gamma, nu_tau, Nitr, thin, similarity, sparse));
    return rcpp_result_gen;
END_RCPP
}
// lcm_pred
SEXP lcm_pred(SEXP X_test, SEXP Y_test, SEXP Group_test, SEXP config_train, int N_test, int S, int C, int K, int G, SEXP itr_draws, SEXP alpha_pi_vec, double alpha_eta, double a_omega, double b_omega, arma::field<arma::cube> lambda_fit, arma::field<arma::cube> phi_fit, arma::cube pi_fit, SEXP pi_init, int Nitr, int similarity);
RcppExport SEXP _LCVA_lcm_pred(SEXP X_testSEXP, SEXP Y_testSEXP, SEXP Group_testSEXP, SEXP config_trainSEXP, SEXP N_testSEXP, SEXP SSEXP, SEXP CSEXP, SEXP KSEXP, SEXP GSEXP, SEXP itr_drawsSEXP, SEXP alpha_pi_vecSEXP, SEXP alpha_etaSEXP, SEXP a_omegaSEXP, SEXP b_omegaSEXP, SEXP lambda_fitSEXP, SEXP phi_fitSEXP, SEXP pi_fitSEXP, SEXP pi_initSEXP, SEXP NitrSEXP, SEXP similaritySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Y_test(Y_testSEXP);
    Rcpp::traits::input_parameter< SEXP >::type Group_test(Group_testSEXP);
    Rcpp::traits::input_parameter< SEXP >::type config_train(config_trainSEXP);
    Rcpp::traits::input_parameter< int >::type N_test(N_testSEXP);
    Rcpp::traits::input_parameter< int >::type S(SSEXP);
    Rcpp::traits::input_parameter< int >::type C(CSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itr_draws(itr_drawsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type alpha_pi_vec(alpha_pi_vecSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_eta(alpha_etaSEXP);
    Rcpp::traits::input_parameter< double >::type a_omega(a_omegaSEXP);
    Rcpp::traits::input_parameter< double >::type b_omega(b_omegaSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::cube> >::type lambda_fit(lambda_fitSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::cube> >::type phi_fit(phi_fitSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type pi_fit(pi_fitSEXP);
    Rcpp::traits::input_parameter< SEXP >::type pi_init(pi_initSEXP);
    Rcpp::traits::input_parameter< int >::type Nitr(NitrSEXP);
    Rcpp::traits::input_parameter< int >::type similarity(similaritySEXP);
    rcpp_result_gen = Rcpp::wrap(lcm_pred(X_test, Y_test, Group_test, config_train, N_test, S, C, K, G, itr_draws, alpha_pi_vec, alpha_eta, a_omega, b_omega, lambda_fit, phi_fit, pi_fit, pi_init, Nitr, similarity));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_LCVA_logsumexp", (DL_FUNC) &_LCVA_logsumexp, 1},
    {"_LCVA_sample_log_prob", (DL_FUNC) &_LCVA_sample_log_prob, 1},
    {"_LCVA_sample_log_prob_matrix_col", (DL_FUNC) &_LCVA_sample_log_prob_matrix_col, 2},
    {"_LCVA_sample_log_prob_matrix", (DL_FUNC) &_LCVA_sample_log_prob_matrix, 1},
    {"_LCVA_sample_prob", (DL_FUNC) &_LCVA_sample_prob, 1},
    {"_LCVA_sample_Dirichlet", (DL_FUNC) &_LCVA_sample_Dirichlet, 1},
    {"_LCVA_lcm_fit", (DL_FUNC) &_LCVA_lcm_fit, 24},
    {"_LCVA_lcm_pred", (DL_FUNC) &_LCVA_lcm_pred, 20},
    {NULL, NULL, 0}
};

RcppExport void R_init_LCVA(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
