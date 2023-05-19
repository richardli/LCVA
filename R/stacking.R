#' Functions implementing stacking computation
#' adapted from codes by Yuling Yao
#' https://github.com/yao-yl/Multimodal-stacking-code
#' 
#' @param log_lik_mat nSample x nChain x n log liklihood array.
#' @param lambda Dirichlet prior
#' @param stack_iter max iteration of optimization
#' @param seed random seed used in generating resampled draws
#' @param nsample number of samples to take
#' @param print_progress whether to print the progress message
#' @import loo rstan
#' @examples
#' 
#' set.seed(12345)
#' loglik <- array(NA, dim = c(1000, 6, 20))
#' for(i in 1:20){
#' 		lik <- runif(1, 0.1, 10)
#' 		for(j in 1:6){
#' 			loglik[, j, i] <- log(lik + runif(1000, -0.01, 0.01))
#' 		}
#' }
#' stacked <- chain_stack(log_lik_mat = loglik, seed = 1, nsample = 500)
#' stacked$draws
#' 
#' @export

chain_stack= function(log_lik_mat, lambda=1.0001, stack_iter=100000, seed = 1, nsample = NULL, print_progress=TRUE  ){

	stacking_weights=function(lpd_point, lambda=1.0001, stack_iter=100000){
		K=dim(lpd_point)[2]
		stacking_opt_stan='
			data {
			  int<lower=0> N;
			  int<lower=0> K;
			  matrix[N,K] lpd_point;
			  vector[K] lambda;
			}
			transformed data{
			  matrix[N,K] exp_lpd_point; 
			  exp_lpd_point=exp(lpd_point);
			}
			parameters {
			   simplex[K] w;
			}
			transformed parameters{
			  vector[K] w_vec;
			  w_vec=w;
			}
			model {
			  for (i in 1: N) {
			    target += log( exp_lpd_point[i,] * w_vec );
			  }
			  w~dirichlet(lambda);
			}
			'
		stan_model_object=stan_model(model_code = stacking_opt_stan) 	
		s_w=optimizing(stan_model_object,  data = list(N=dim(lpd_point)[1], K=K, lpd_point=lpd_point, lambda=rep(lambda, dim(lpd_point)[2])), iter=stack_iter)$par[1:K] 
		return(s_w)
	} 

	mixture_draws= function (S, K,  weight, random_seed=1, permutation=TRUE){

		set.seed(random_seed)
		integer_part=floor(S*weight)
		existing_draws=sum(integer_part)
		if(existing_draws<S){
			remaining_draws=S-existing_draws
			update_w=(weight- integer_part/S)*  S / remaining_draws
			remaining_assignment=sample(1:K, remaining_draws, prob =update_w , replace = F)
			integer_part[remaining_assignment] =integer_part[remaining_assignment]+1
		}
		integer_part_index=c(0,cumsum(integer_part))
		mixture_vector=rep(NA, S)
		
		return(integer_part)
	}



	n= dim(log_lik_mat)[3]
	K= dim(log_lik_mat)[2]
	S= dim(log_lik_mat)[1]
	if(print_progress==TRUE){
		cat(paste("Stacking", K, "chains, with",n, "data points and", S,  "posterior draws;\n using stan optimizer, max iterations =",stack_iter ,"\n" ))
		sysTimestamp=Sys.time()
	}
	loo_elpd=matrix(NA,n, S)
	options(warn=-1)
	loo_chain=apply(log_lik_mat, 2, function(lp){
		loo_obj=loo::loo(lp)
		return(c(loo_obj$pointwise[,1], loo_obj$diagnostics
						 $pareto_k ))  
	}) 
	options(warn=0)
	loo_elpd= loo_chain[1:n, ]
	chain_weights=stacking_weights(lpd_point=loo_elpd, lambda=lambda, stack_iter=stack_iter)
	pareto_k=loo_chain[(n+1):(2*n), ]
	if(print_progress==TRUE){
		cat(paste("\n Total elapsed time for approximate LOO and stacking =", round(Sys.time()
	-sysTimestamp,digits=2), "s\n" ))
		}
	if(is.null(nsample)) nsample <- dim(log_lik_mat)[1]
	draws <- mixture_draws(S = nsample, K = dim(log_lik_mat)[2], weight = chain_weights, random_seed = seed)
	return( list(chain_weights=chain_weights, pareto_k=pareto_k, draws = draws ))
}
 
