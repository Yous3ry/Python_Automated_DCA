# load Sample well data
df <- read.csv(file = 'Sample_Well.csv', header=TRUE)
df$Date <-as.Date(df$Date, format = "%m/%d/%Y")
df.index <- df$Date
head(df)

# convert data to time series
oil_ts <- ts(data=df[,2], frequency=12)
oil_ts

# Change points library
library(EnvCpt)
x <- as.integer()
# define models to be used to find Change points
models <- c("trendar1cpt", "trendar2cpt")
# Find Change points & Limit Minimum Period length to 5% of Well History
out <- envcpt(oil_ts, verbose=0, models=models, minseglen=as.integer(0.05*length(df.index),0))
# Note you can change Min length as needed
AIC(out)
plot(out, type="fit")
# find minimum AIC model
optimal_model <- which.min(AIC(out))
optimal_model <- names(optimal_model[1])
optimal_model
# Find change points in model
change_points <- cpts(out[[optimal_model]])
change_points

# plot results
plot(df, log='y')
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 1)      # Grid line width
abline(v = df$Date[change_points], col="red", lwd=2)
