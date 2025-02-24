#' function to extract the predictive probabilities for a new data matrix
#' 
#' @param fit  and LCVA.pred object
#' @param X a matrix of dimension n by p, in the same format as the input to LCVA.pred
#' @param Burn_in number of iterations to discard
#' 
#' @return an Nitr x N x C matrix of posterior draws of P(Y | X)
#' @export
#' 
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, 
#' 				Y = simLCM$Y_train, Domain = simLCM$G_train, 
#'              causes.table = simLCM$causes.table, 
#'              domains.table = simLCM$domains.table,
#'            K = 4, model = "M", Nitr = 400, nchain = 5, seed = 1234)
#' 
#' out <- LCVA.pred(fit = out.train, X_test = simLCM$X_test,  
#'                  model = "D", 
#'                  alpha_pi = 1, alpha_eta = .1, 
#'                  Burn_in_train = 200, Nitr = 200)
#' out1 <- LCVA.pred(fit = out.train, X_test = simLCM$X_test,  
#'                  model = "D", 
#'                  alpha_pi = 1, alpha_eta = .1, 
#'                  Burn_in_train = 200, Nitr = 200, return_likelihood = TRUE)
#' Yhat <- get_assignment(out$Y_test, Burn_in = 100)
#' # Posterior predictive prob is an Nitr * N * C array 
#' prob <- get_pred_prob(out, simLCM$X_test, Burn_in = 100)
#' prob_mean <- apply(prob, c(2, 3), mean)
#' Yhat_pred <- apply(prob_mean, 1, which.max)  
#' # The two should be quite similar
#' table(Yhat, Yhat_pred)
#' 
#' }

get_pred_prob <- function(fit, X, Burn_in = NULL){
	Nitr <- length(fit$loglambda)
	if(is.null(Burn_in)) Burn_in <- 0
	N <- dim(X)[1]
	C <- dim(fit$pi_test)[2]
	P <- dim(X)[2]
	K <- dim(fit$loglambda[[1]])[2]

	prob_out <- array(NA, c(Nitr - Burn_in, N, C))
	for(itr in (Burn_in + 1) : Nitr){
		# C vector
		pi <- fit$pi_test[itr, ] 
		# C * K * p
		phi <- fit$phi[[itr]]
		# C * K
		lambda <- exp(fit$loglambda[[itr]])

		prob <- array(1, c(N, C, K))
		# Prob_{ic} = pi_c * \sum_k lambda[c, k] ( \prod_j  phi[c, k, j] ^ x[, j] * (1 -phi)^(1-x) ) 
		for(c in 1:C){
			for(k in 1:K){
				for(j in 1:P){
					tmp <- phi[c, k, j] ^ X[, j] * (1 - phi[c, k, j]) ^ (1 - X[, j]) 
					tmp[is.na(tmp)] <- 1
					prob[ , c, k] <- prob[, c, k] * tmp 
				}
				prob[ , c, k] <- prob[ , c, k] * lambda[c, k]
			}
			prob[ , c, k] <- prob[ , c, k] * pi[c]
		}
		tmpprob <- apply(prob, c(1, 2), sum)
		tmpprob <- tmpprob / apply(tmpprob, 1, sum)
		prob_out[itr - Burn_in, , ] <- tmpprob
	}
	return(prob_out)
}