library(FITSio)
library(reshape2)
library(dplyr)
library(tidyr)
library(ggplot2)


# Functions:

normalize_asinh <- function(x, pmin = 10, pmax = 99.9, scale = 0.02) {
  lo <- quantile(x, pmin/100, na.rm = TRUE)
  hi <- quantile(x, pmax/100, na.rm = TRUE)
  
  y <- (x - lo) / (hi - lo)
  y[y < 0] <- 0
  y[y > 1] <- 1
  
  y <- asinh(y / scale) / asinh(1 / scale)
  
  y[y < 0] <- 0
  y[y > 1] <- 1
  
  return(y)
}


normalize_asinh_background <- function(x, pmin = 10, pmax = 99.9, scale = 0.1, floor = 0.05) {
  lo <- quantile(x, pmin/100, na.rm = TRUE)
  hi <- quantile(x, pmax/100, na.rm = TRUE)

  y <- (x - lo) / (hi - lo)
  y[y < 0] <- 0
  y[y > 1] <- 1

  y <- asinh(y / scale) / asinh(1 / scale)

  y[y < floor] <- 0
  y[y > 1] <- 1

  return(y)
}


subtract_background <- function(x) {
  bkg <- median(x, na.rm = TRUE)
  x - bkg
}


make_rgb_image <- function(r_path, g_path, b_path,
                           out_path = "rgb_output.png",
                           normalize_fn = NULL,
                           background_fn = NULL) {
  
  library(FITSio)
  library(ggplot2)
  library(dplyr)
  
  # Read the FITS files
  r_band <- readFITS(r_path)
  g_band <- readFITS(g_path)
  b_band <- readFITS(b_path)
  
  R <- r_band$imDat
  G <- g_band$imDat
  B <- b_band$imDat
  
  # Check dimensions
  stopifnot(
    all(dim(R) == dim(G)),
    all(dim(R) == dim(B))
  )
  
  # Optional background subtraction
  if (!is.null(background_fn)) {
    R <- background_fn(R)
    G <- background_fn(G)
    B <- background_fn(B)
  }
  
  # Default normalization: asinh + percentis + floor
  if (is.null(normalize_fn)) {
    normalize_fn <- function(x, pmin = 5, pmax = 99, scale = 0.02, floor = 0.02) {
      lo <- quantile(x, pmin/100, na.rm = TRUE)
      hi <- quantile(x, pmax/100, na.rm = TRUE)
      
      if (hi == lo) {
        return(matrix(0, nrow = nrow(x), ncol = ncol(x)))
      }
      
      # linear normalization
      y <- (x - lo) / (hi - lo)
      y[y < 0] <- 0
      y[y > 1] <- 1
      
      # asinh stretch
      y <- asinh(y / scale) / asinh(1 / scale)
      
      # remove background pedestal
      y[y < floor] <- 0
      y[y > 1] <- 1
      
      return(y)
    }
  }
  
  # Normalize
  Rnorm <- normalize_fn(R)
  Gnorm <- normalize_fn(G)
  Bnorm <- normalize_fn(B)

  # Weight channels
  wR <- 1.02
  wG <- 0.98
  wB <- 0.98

  Rnorm <- Rnorm * wR
  Gnorm <- Gnorm * wG
  Bnorm <- Bnorm * wB

  Rnorm[Rnorm > 1] <- 1
  Gnorm[Gnorm > 1] <- 1
  Bnorm[Bnorm > 1] <- 1
  
  # Build RGB dataframe (correct pixel order)
  df <- data.frame(
    x = rep(1:ncol(R), times = nrow(R)),
    y = rep(nrow(R):1, each = ncol(R)),
    r = as.vector(t(Rnorm)),
    g = as.vector(t(Gnorm)),
    b = as.vector(t(Bnorm))
  ) %>%
    mutate(rgb = rgb(r, g, b))
  
  # Plot
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_raster(aes(fill = rgb)) +
    scale_fill_identity() +
    coord_fixed() +
    theme_void()
  
  # Save
  ggsave(out_path, plot = p, width = 8, height = 8, dpi = 300)
  
  print(p)
  return(p)
}


# How to use:

# p <- make_rgb_image(
#  r_path = "F200W.fits",
#  g_path = "F115W.fits",
#  b_path = "F090W.fits",
#  out_path = "RGB.png", 
#  normalize_fn = normalize_asinh,  # or normalize_asinh_background
#  background_fn = subtract_background)
