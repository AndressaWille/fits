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


make_rgb_image_scale <- function(r_path, g_path, b_path,
                           out_path = "rgb_output.png",
                           normalize_fn = NULL,
                           background_fn = NULL,
                           scale_pix = NULL,
                           scale_label = "10 kpc",
                           tail = NULL) {
  
  library(FITSio)
  library(ggplot2)
  library(dplyr)
  
  R <- readFITS(r_path)$imDat
  G <- readFITS(g_path)$imDat
  B <- readFITS(b_path)$imDat
  
  stopifnot(all(dim(R) == dim(G)), all(dim(R) == dim(B)))
  
  if (!is.null(background_fn)) {
    R <- background_fn(R)
    G <- background_fn(G)
    B <- background_fn(B)
  }
  
  if (is.null(normalize_fn)) {
    normalize_fn <- function(x, pmin = 5, pmax = 99, scale = 0.02, floor = 0.02) {
      lo <- quantile(x, pmin/100, na.rm = TRUE)
      hi <- quantile(x, pmax/100, na.rm = TRUE)
      y <- (x - lo) / (hi - lo)
      y[y < 0] <- 0; y[y > 1] <- 1
      y <- asinh(y / scale) / asinh(1 / scale)
      y[y < floor] <- 0
      y
    }
  }
  
  Rn <- normalize_fn(R)
  Gn <- normalize_fn(G)
  Bn <- normalize_fn(B)
  
  df <- data.frame(
    x = rep(1:ncol(R), times = nrow(R)),
    y = rep(nrow(R):1, each = ncol(R)),
    r = as.vector(t(Rn)),
    g = as.vector(t(Gn)),
    b = as.vector(t(Bn))
  ) %>%
    mutate(rgb = rgb(r, g, b))
  
  p <- ggplot(df, aes(x = -y, y = x)) +
    geom_raster(aes(fill = rgb)) +
    scale_fill_identity() +
    coord_fixed() +
    theme_void()
  
  # Bar scale (how many pixels are equivalent to 10 kpc?)
  if (!is.null(scale_pix)) {
    
    x_min <- min(-df$y)
    x_max <- max(-df$y)
    y_min <- min(df$x)
    y_max <- max(df$x)
    
    x_start <- x_max - scale_pix - 0.05 * diff(range(-df$y))
    x_end   <- x_start + scale_pix
    y_bar   <- y_min + 0.05 * diff(range(df$x))
    
    p <- p +
      geom_segment(
        aes(x = x_start, xend = x_end,
            y = y_bar,   yend = y_bar),
        linewidth = 1.2,
        colour = "white"
      ) +
      annotate(
        "text",
        x = (x_start + x_end) / 2,
        y = y_bar + 0.03 * diff(range(df$x)),
        label = scale_label,
        colour = "white",
        size = 6
      )
  }
  
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

# p <- make_rgb_image_scale(
#  r_path = "F200W.fits",
#  g_path = "F115W.fits",
#  b_path = "F090W.fits",
#  out_path = "RGB_with_scale.png", 
#  normalize_fn = normalize_asinh, # or normalize_asinh_background
#  background_fn = subtract_background,
#  scale_pix = 110, # example: 10 kpc are equivalent to 110 pixels
#  scale_label = "10 kpc")

