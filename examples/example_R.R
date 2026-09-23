# example_R.R -- run after: R CMD INSTALL --preclean .  (from the R/ folder)
library(smop)

set.seed(0)
m <- 40; n <- 150
A <- matrix(rnorm(m * n), m, n)
xtrue <- rep(0, n); xtrue[sample(n, 8)] <- rnorm(8)
b <- A %*% xtrue + 0.02 * rnorm(m)
delta <- 1.02 * norm(A %*% xtrue - b, "2")

res <- solve_bmop(A, b, delta, options = list(verbose = 0))
cat("status:", res$status, res$msg, "\n")
cat("||Ax-b|| =", norm(A %*% res$x - b, "2"), " delta =", delta, "\n")
cat("||x||_1 =", sum(abs(res$x)), " (true", sum(abs(xtrue)), ")\n")
cat("iter:", res$iter, " bisection:", res$iter_bisection,
    " secant/newton:", res$iter_newton_or_secant, "\n")
cat("support:", sum(abs(res$x) > 1e-8), "(true 8)\n")

# single Lasso
lr <- solve_lasso(A, b, 0.05)
cat("lasso obj:", lr$obj, " kkt eta:", lr$eta, "\n")
