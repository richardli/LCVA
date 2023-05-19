#' function to extract the most likely cause of death from posterior draws
#' 
#' @param draws nSample by n matrix
#' @param Burn_in number of iterations to discard
#' @export
#' 
#' @examples
#' \dontrun{
#' data(simLCM)
#' out.train <- LCVA.train(X = simLCM$X_train, 
#' 				Y = simLCM$Y_train, Domain = simLCM$G_train, 
#'              causes.table = simLCM$causes.table, 
#'              domains.table = simLCM$domains.table,
#'            K = 5, model = "M", Nitr = 400, nchain = 5, seed = 1234)
#' out <- LCVA.pred(fit = out.train, X_test = simLCM$X_test,  
#'                  model = "D", 
#'                  alpha_pi = 1, alpha_eta = .1, 
#'                  Burn_in_train = 200, Nitr = 200)
#' Yhat <- get_assignment(out$Y_test, Burn_in = 100)
#' sum(simLCM$Y_test == Yhat) / length(Yhat)
#' }

get_assignment <- function(draws, Burn_in = NULL){
	if(!is.null(Burn_in)){
		Ypred <- draws[-c(1:Burn_in), ]
	}else{
		Ypred <- draws
	}
	Yhat <- apply(Ypred, 2, function(x){as.numeric(names(sort(table(x), decreasing=TRUE))[1])})
	return(Yhat)
}