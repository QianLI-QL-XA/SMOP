#======================================================================
# smop.R -- R-level wrappers for the smop C++ core (Rcpp bindings)
#======================================================================

#' Solve the BMOP (Basis-pursuit with a level-set constraint):
#'   min ||x||_1  s.t.  ||A x - b||_2 <= delta
#' by the level-set method (secant / Newton / bisection + adaptive sieving).
#'
#' @param A      an m x n numeric matrix (dense)
#' @param b      numeric vector of length m
#' @param delta  positive scalar constraint radius
#' @param options optional named list, e.g.
#'   list(stoptol=1e-6, maxiter=200, use_secant=TRUE, use_as=TRUE, verbose=0)
#' @param x0     optional warm start vector
#' @return a list with elements x, xi, y, mu, psi, eta, iter, status, msg,
#'   time, mupath, psipath, reducedn
#' @export
solve_bmop <- function(A, b, delta, options = NULL, x0 = NULL) {
  if (!is.matrix(A)) A <- as.matrix(A)
  b <- as.numeric(b)
  stopifnot(ncol(A) == length(b) || nrow(A) == length(b))
  if (nrow(A) != length(b)) A <- t(A)          # column convention
  smop_solve_bmop(A, b, delta, options, x0)
}

#' Solve the Lasso subproblem:
#'   min 1/2 ||A x - b||_2^2 + lambda ||x||_1
#' @export
solve_lasso <- function(A, b, lambda, options = NULL, x0 = NULL) {
  if (!is.matrix(A)) A <- as.matrix(A)
  b <- as.numeric(b)
  stopifnot(ncol(A) == length(b) || nrow(A) == length(b))
  if (nrow(A) != length(b)) A <- t(A)
  smop_solve_lasso(A, b, lambda, options, x0)
}

#' Package version.
#' @export
smop_version <- function() smop_version_cpp()
