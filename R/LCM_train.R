#' Fit the nested latent class model on training data
#' 
#' @param X a n by p matrix of the symptoms with 0 being absent, 1 being present, and NA being missing.
#' @param Y a vector of length n of the causes-of-death. It can be one of the two formats: (1) a vector of cause-of-death label; (2) a numeric vector taking values from 1 to C, where C is the total number of all cause. 
#' @param Domain a vector of length n of the domain indicators, coded into 1 to G, where G is the total number of all domains.
#' @param K number of latent classes within each cause-of-death
#' @param model the model to be fit. The current choice of models include Single-domain ("S") or multi-domain model ("M").
#' @param Nitr number of iterations to run in each MCMC chain.
#' @param nchain number of MCMC chains to run.
#' @param seed a vector of seeds to be used for each MCMC chain.
#' @param causes.table a vector of causes-of-death labels.
#' @param domains.table  a vector of training domains labels. Not including the target domain.
#' @param alpha_pi Concentration parameter for the training domain CSMF prior. Default to be 1
#' @param a_omega Shape parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior. Can be either a scalar or a C by p matrix. Default to be 1
#' @param b_omega Rate parameter of the gamma distribution for the omega_c parameter in the stick-breaking prior. Can be either a scalar or a C by p matrix.  Default to be 1
#' @param nu_phi Shape2 parameter of the beta distribution for the class-dependent response probabilities. Default to be 1
#' @param nu_tau Shape2 parameter of the sparsity level in the response probabilities. Default to be 1
#' @param a_base Shape1 parameter of the gamma prior for the baseline response probabilities. Default to be 1
#' @param b_base Shape2 parameter of the gamma prior for the baseline response probabilities. Default to be 1

#' 
#' @return a fitted object in the class of LCVA
#' @importFrom stats median quantile sd
#' @import RcppArmadillo
#' @export
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, Y = simLCM$Y_train, 
#' 						Domain = simLCM$G_train, 
#' 			   			causes.table = simLCM$causes.table, 
#' 					    domains.table = simLCM$domains.table,
#' 						K = 5, model = "M", Nitr = 400, nchain = 1, 
#' 						seed = 1234)
#' summary(out.train)
#' }
LCVA.train <- function(X, Y, Domain, K, model = NULL, Nitr = 4000, nchain = 1, 
					  seed = NULL, causes.table = NULL, domains.table = NULL, 
					  alpha_pi = 1, a_omega = 1, b_omega = 1, 
					  nu_phi = 1, nu_tau = 1, a_base = 1, b_base = 1){

	if(is.null(model) || model %in% c("S", "M") == FALSE){
		stop("'model' needs to be one of the following: S, M")
	}

	if(is.null(causes.table)){
		causes.table <- sort(unique(Y))
	}
	if(is.null(domains.table)){
		domains.table <- sort(unique(Domain))
	}
	# todo: make input a data frame?
	S <- dim(X)[2]
	C <- length(causes.table)
	G <- length(domains.table)

	if(!is(Y, "numeric")){
		Y <- match(Y, causes.table)
		if(sum(is.na(Y)) > 0){
			stop("There exist Y that are not in the `causes.table` vector.")
		}
	}
	if(!is(Domain, "numeric")){
		Domain <- match(Domain, domains.table)
		if(sum(is.na(Domain)) > 0){
			stop("There exist Domain that are not in the `domains.table` vector.")
		}
	}

	similarity = 1
	G.fit <- G
	if(model == "S"){
		G_train <- rep(1, length(Domain))
		G.fit <- 1
	}
	
	if(is.null(seed)){
		seeds <- sample(1:1e4, nchain)
	}else if(length(seed) < nchain){
		seed <- seed[1] + c(0 : (nchain - 1))
	}else if(length(seed) > nchain){
		seed <- seed[1:nchain]
	}

	if(length(a_base) == 1){
		a_base <- matrix(a_base, C, S)
	}
	if(length(b_base) == 1){
		b_base <- matrix(b_base, C, S)
	}


	all.train <- NULL
	index <- 1
	for(seed.train in seed){
		set.seed(seed.train)		

		t0 <- Sys.time()
		# a_omega and b_omega are shape and RATE here
		out.train <- lcm_fit(X = X, Y = Y, Group = Domain, N_train = dim(X)[1],   
			  	   X_test = matrix(NA), Y_test = NA, Group_test = NA, N_test = 0,
				   S = S, C = C, G = G, K = K, 
				   alpha_pi = alpha_pi, alpha_eta = 1, a_omega = a_omega, b_omega = b_omega, 
				   nu_phi = nu_phi, nu_tau = nu_tau, 
				   a_gamma = a_base, b_gamma = b_base, 
				   Nitr = Nitr, thin = 1, similarity = similarity, sparse = 1)  
		t1 <- Sys.time()
		print(t1 - t0)
		out.train$time <- t1 - t0
		out.train$seed <- seed.train
		all.train[[index]] <- out.train
		all.train[[index]]$model <- model
		all.train[[index]]$similarity <- similarity
		all.train[[index]]$Nitr <- Nitr
		all.train[[index]]$causes.table <- causes.table
		all.train[[index]]$domains.table <- domains.table
		index <- index + 1
	}

	class(all.train) <- "LCVA"
	return(all.train)
}


#' Summary method for the fitted model from \code{LCVA.train}.
#' 
#' This function is the summary method for class \code{LCVA}.
#' 
#' 
#' @param object output from \code{\link{LCVA.train}} 
#' @param ... not used
#' 
#' @seealso \code{\link{LCVA.train}} 
#' @method summary LCVA
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, Y = simLCM$Y_train, 
#' 						Domain = simLCM$G_train, 
#' 			   			causes.table = simLCM$causes.table, 
#' 					    domains.table = simLCM$domains.table,
#' 						K = 5, model = "MD", Nitr = 400, nchain = 1, 
#' 						seed = 1234)
#' summary(out.train)
#' }
#' @export 

summary.LCVA <- function(object,...){
	sd <- median <- quantile <- NULL
	nchain <- length(object)
	C <- object[[1]]$C
	S <- object[[1]]$S
	K <- object[[1]]$K
	G <- object[[1]]$G
	Nitr <- object[[1]]$Nitr
	Model <- object[[1]]$model # c("S", "SN", "MN", "MD", "MDC")
	if(Model == "S"){
		Model <- "Single-domain model with constant mixing weight"
	}else if(Model == "SN"){
		Model <- "Single-domain model with new mixing weight"
	}else if(Model == "MN"){
		Model <- "Multi-domain model with new mixing weight"
	}else if(Model == "MD"){
		Model <- "Multi-domain model with domain-level mixture"		
	}else if(Model == "MDC"){
		Model <- "Multi-domain model with domain-cause-level mixture"
	} 

	cat("----------------------------------\n")
	cat(Model)
	cat("\nModel trained on ")
	cat(G)
	cat(" domains\n") 
	cat(nchain)
	cat(" chain(s) constructed\n") 
	cat(Nitr)
	cat(" iterations of posterior samples drawn in each chain\n")  
	cat("----------------------------------\n")
	
}

