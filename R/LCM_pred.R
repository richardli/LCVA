#' Fit the latent class model
#' 
#' @param fit a fitted LCVA object.
#' @param X_test a n by p matrix of the symptoms with 0 being absent, 1 being present, and NA being missing.
#' @param Y_test a vector of length n of the causes-of-death. It can be one of the two formats: (1) a vector of cause-of-death labels as in the training data, where NA or any labels not in the cause list is treated as unknown causes of death; (2) a numeric vector taking values from 0 to C, where C is the total number of all cause and 0 indicates unknown cause of death. Default to be NULL, which is treated as all causes unknown.
#' @param Domain_test a vector of length n of the domain indicators, coded into 0 to G, where G is the total number of all domains and 0 indicates a new target domain. Default to be NULL, which is treated as all deaths from target domain.
#' @param model prediction model for the new domain. Current available choices are: constant mixing weight ("C", only applicable to single domain training models), new mixing weights ("N"), domain-level mixture ("D", only applicable to multi-domain training models), domain-cause level mixture ("DC", only applicable to multi-domain training models)
#' @param alpha_pi concentration parameter for the target domain CSMF prior. If left unspecified, it will be set to the number of training deaths per cause multiplied by 0.1.
#' @param alpha_eta concentration parameter for the domain or domain-cause mixture prior. Only used for multi-domain models. Default to be 1.
#' @param Nitr number of posterior draws to save. If set to NULL, it will be set to the total number of training MCMC samples.
#' @param Burn_in_train number of posterior draws to discard as burn-in in each training chain. If set to NULL, first half of each training chain will be discarded. 
#' @param return_likelihood whether the P(X|Y) is returned for each of the unlabeled observations. If this is set to TRUE, the result will contain an array of posterior draws of size Nitr * N * C array of P(X_i | Y = c) for i = 1, ..., N, and c = 1, ..., C.  
#' @param verbose logical indicator of whether to print out the MCMC progress.
#' 
#' @return a fitted object in the class of LCVA.pred
#' @importFrom methods is
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, 
#'              Y = simLCM$Y_train, Domain = simLCM$G_train, 
#'              causes.table = simLCM$causes.table, 
#'              domains.table = simLCM$domains.table,
#'              K = 5, model = "M", Nitr = 400, 
#'              nchain = 5, seed = 1234)
#' out <- LCVA.pred(fit = out.train, X_test = simLCM$X_test, 
#'                  model = "D", 
#'                  alpha_pi = 1, alpha_eta = .1, 
#'                  Burn_in_train = 200, Nitr = 200)
#' # Compare with the true fractions
#' summary(out, Burn_in = 100)
#' sort(table(simLCM$Y_test)/length(simLCM$Y_test), dec = TRUE)
#'
#' # Compare individual results
#' Yhat <- get_assignment(out$Y_test, Burn_in = 100)
#' sum(simLCM$Y_test == Yhat) / length(Yhat)
#' }

LCVA.pred <- function(fit, X_test, Y_test = NULL, Domain_test = NULL, 
                      model = NULL,
                      alpha_pi = NULL, alpha_eta = 1, 
                      Nitr = NULL, Burn_in_train = NULL, return_likelihood = FALSE, verbose = TRUE){

  if(!methods::is(fit, "LCVA")){
    stop("The argument 'fit' needs to be of class LCVA.")
  }
  if(fit[[1]]$model == "M" && model == "C"){
    stop("model = 'C' is only compatible with single domain models, i.e., LCVA.train(model = 'S')")
  }
  if(fit[[1]]$model == "S" && model == "D"){
    stop("model = 'D' is only compatible with multi-domain models, i.e., LCVA.train(model = 'M')")
  }
  if(fit[[1]]$model == "S" && model == "DC"){
    stop("model = 'DC' is only compatible with multi-domain models, i.e., LCVA.train(model = 'M')")
  }

  if(!is.null(Y_test) && !is(Y_test, "numeric")){
    Y_test <- match(Y_test, fit[[1]]$causes.table)
    Y_test[is.na(Y_test)] <- 0
  }else if(!is.null(Y_test)){
    Y_test[is.na(Y_test)] <- 0    
  }


  if(is.null(Burn_in_train)) Burn_in_train <- round(fit[[1]]$Nitr / 2)

  # Get the domain-cause counts
  if(is.null(fit[[1]]$config)){
    config <- matrix(0, fit[[1]]$G, fit[[1]]$C)
    for(g in 1:fit[[1]]$G){
      for(c in 1:fit[[1]]$C){
        config[g, c] <- sum(fit[[1]]$Group == g & fit[[1]]$Y == c) / sum(fit[[1]]$Y == c)
      }
    }
  }else{
    config <- fit[[1]]$config
  }

  if(length(fit) > 1 & model == "N"){
    fit0 <- fit 
    fit <- NULL 
    fit[[1]] <- fit0[[1]]
    message("Single domain model with new weights fitted, only the first MCMC chain is used.")
  } 
  
  if(length(fit) > 1){
      Nitr.train <- length(fit[[1]]$loglambda) 
      sub <- (Burn_in_train + 1) : Nitr.train
      loglik <- array(NA, dim = c(length(sub), length(fit), dim(fit[[1]]$loglik)[2]))
      for(j in 1:dim(loglik)[2]){
          loglik[, j, ] <- fit[[j]]$loglik[sub, ] 
      }
      resample_model <- chain_stack(log_lik_mat = loglik, seed = 1, nsample = Nitr, print_progress = TRUE)
      resample <- resample_model$draws
       print("Number of posterior draws from each chain:")
       print(resample)    
  }else{
    resample <- Nitr
    Nitr.train <- length(fit[[1]]$loglambda) 
  }

  loglambda <- phi <- gamma <- NULL
  pi_fit <- array(NA, dim = c(sum(resample), dim(fit[[1]]$pi)[2], dim(fit[[1]]$pi)[3]))
  counter <- 0
  for(i in 1:length(fit)){
    tmp <- sample(c(Burn_in_train + 1): Nitr.train, resample[i], replace = TRUE)
    for(j in tmp){
      counter <- counter + 1
      loglambda[[counter]] <- fit[[i]]$loglambda[[j]]
      phi[[counter]] <- fit[[i]]$phi[[j]]
      gamma[[counter]] <- fit[[i]]$gamma[[j]]
      pi_fit[counter, , ] <- fit[[i]]$pi[j, , ]
    }
  }
  
  itr_draws <- sample(1:Nitr) - 1


  if(is.null(alpha_pi)){
    alpha_pi <- 1 #length(Y_test) / fit[[1]]$C * .1
  }
  if(length(alpha_pi) == 1){
    alpha_pi <- rep(alpha_pi, fit[[1]]$C)
  }
  if(is.null(Y_test)){
    Y_test <- rep(0, dim(X_test)[1])
  }
  if(is.null(Domain_test)){
    Domain_test <- rep(0, dim(X_test)[1])
  }
  if(length(alpha_pi) != fit[[1]]$C){
    stop(paste("The argument 'alpha_pi' needs to be a vector of length", fit[[1]]$C))
  }
  pi_init <- rep(1/fit[[1]]$C, fit[[1]]$C)
  csmf0 <- rep(1, fit[[1]]$C)
  # if(info_prior){
  #    csmf0 <- as.numeric(table(c(1:fit[[1]]$C, fit[[1]]$Y[,1])) - 1)  
  #    csmf0 <- csmf0 / sum(csmf0) * fit[[1]]$C
  # }

  # for model = N
  similarity = 0
  if(model == "C"){
    similarity = 1
  }else if(model == "D"){
    similarity = 1
  }else if(model == "DC"){
    similarity = 2
  }
  t0 <- Sys.time()
  out <- lcm_pred(
            X_test = X_test, 
            Y_test = Y_test, 
            Group_test = Domain_test, 
            config_train = config,
            N_test = dim(X_test)[1], 
            S = fit[[1]]$S, 
            C = fit[[1]]$C,  
            G = fit[[1]]$G, 
            K = fit[[1]]$K, 
            itr_draws = itr_draws,
            alpha_pi_vec = alpha_pi * csmf0, 
            alpha_eta = alpha_eta, 
            a_omega = fit[[1]]$a_omega, 
            b_omega = fit[[1]]$b_omega, 
            lambda_fit = loglambda, 
            phi_fit = phi, 
            pi_fit = pi_fit,
            pi_init = pi_init,
            Nitr = Nitr, 
            # Burn_in = 0,
            similarity = similarity, 
            return_x_given_y = as.integer(return_likelihood),
            verbose = as.integer(verbose)) 
  t1 <- Sys.time()
  
  out$phi <- phi
  out$loglambda.train <- loglambda
  out$gamma <- gamma 
  out$causes.table <- fit[[1]]$causes.table
  out$domains.table <- fit[[1]]$domains.table
  out$model.train <- fit[[1]]$model
  out$model.test <- model
  out$Nitr <- Nitr 
  out$Nitr.train <- Nitr.train
  out$Ntrain <- length(fit)
  out$time <- t1 - t0

  class(out) <- "LCVA.pred"
  return(out)
}


#' Summary method for the fitted model from \code{LCVA.pred}.
#' 
#' This function is the summary method for class \code{LCVA.pred}.
#' 
#' 
#' @param object output from \code{\link{LCVA.pred}} 
#' @param CI credible interval to show in the summary
#' @param Burn_in number of initial iterations to discard as burn-in in the prediction stage.
#' @param top number of top causes to display.
#' @param ... not used
#' 
#' @seealso \code{\link{LCVA.pred}} 
#' @method summary LCVA.pred
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, Y = simLCM$Y_train, Domain = simLCM$G_train, 
#'              causes.table = simLCM$causes.table, 
#'              domains.table = simLCM$domains.table,
#'            K = 5, model = "M", Nitr = 400, nchain = 1, seed = 1234)
#' out <- LCVA.pred(fit = out.train, X_test = simLCM$X_test,  
#'                  model = "D",
#'                  alpha_pi = 1, alpha_eta = .1, 
#'                  Burn_in_train = 200, Nitr = 200)
#' summary(out)
#' # Compare with the truth
#' sort(table(simLCM$Y_test)/length(simLCM$Y_test), dec = TRUE)
#' }
#' @export 

summary.LCVA.pred <- function(object, CI = 0.95, Burn_in = NULL, top = 5, ...){
  nchain <- object$Ntrain
  S <- object$S
  K <- object$K
  G <- object$G
  Nitr <- object$Nitr
  Nitr.train <- object$Nitr.train
  Model <- object$model.test # c("SC", "SN", "MN", "MD", "MDC")
  if(Model == "C"){
    Model <- "Constant mixing weight in target domain"
  }else if(Model == "N"){
    Model <- "New independent mixing weight in target domain"
  }else if(Model == "D"){
    Model <- "Domain-level mixture of mixing weights in target domain"   
  }else if(Model == "DC"){
    Model <- "Domain-cause-level mixture of mixing weights in target domain"
  } 

  if(is.null(Burn_in)) Burn_in <- round(Nitr / 2)
  pi_test <- object$pi_test[(Burn_in + 1) : Nitr, ]
  csmf <- data.frame(Mean = apply(pi_test, 2, mean), 
                     SD = apply(pi_test, 2, sd), 
                     Median = apply(pi_test, 2, median), 
                     Lower = apply(pi_test, 2, quantile, (1-CI)/2), 
                     Upper = apply(pi_test, 2, quantile, 1 - (1-CI)/2)
                     )
  rownames(csmf) <- object$causes.table
  csmf.top <- csmf[order(csmf[,1], decreasing = TRUE)[1:min(c(top, dim(csmf)[1]))],]
  csmf.top <- round(csmf.top, 4)

  cat("----------------------------------\n")
  cat(Model)
  cat("\nModel trained on ")
  cat(length(out$domains.table))
  cat(" domains\n")  
  cat(nchain)
  cat(" chain(s) constructed\n") 
  cat(Nitr.train)
  cat(" iterations of posterior samples drawn in each training chain\n")  
  cat(Nitr)
  cat(" iterations of posterior samples drawn in the prediction stage\n")  
  cat("----------------------------------\n")
  print(csmf.top)
  cat("----------------------------------\n")
  
}
